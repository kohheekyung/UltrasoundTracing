/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "raytracing.h"
#include <cuComplex.h>
#define CUDART_PI_F 3.141592654f


//ray
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(PerRayData_beam, prd_beam, rtPayload, );
rtDeclareVariable(optix::Ray,  ray,          rtCurrentRay, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float, refraction_index, , );
rtDeclareVariable(float3, target, , );


//properties
rtDeclareVariable(float, water_density, , );
rtDeclareVariable(float, water_speed, , );
rtDeclareVariable(float, skull_density, , );
rtDeclareVariable(float, skull_speed, , );
rtDeclareVariable(float, random_density, , );
rtDeclareVariable(float, random_speed, , );

//output buffer
rtBuffer<double, 2> out_rc_buffer;
rtBuffer<double, 2> test_rc_buffer;
rtBuffer<float3, 2> out_intersection_point;
rtBuffer<float3, 2> in_intersection_point;
rtBuffer<float3, 2> reflec_in_skull2out;

//input buffer
rtBuffer<float3>        origins;

using namespace optix;

__device__ __forceinline__ cuDoubleComplex _exp(cuDoubleComplex z)
{
	cuDoubleComplex res;
	float t = expf(z.x);
	sincos(z.y, &res.y, &res.x);
	res.x *= t;
	res.y *= t;
	return res;
}

RT_PROGRAM void ray_generation()
{
	float lambda = 1000 * skull_speed / 200000;
	float k = 2 * CUDART_PI_F / lambda;

	double angle_outskull = 0.0;
	double angle_inskull = 0.0;
	double angle_reflection = 0.0;
	double out_rc;
	double in_rc;
	double reflect_rc;
	double test_rc;
	double RC_water_skull;
	double TC_water_skull;
	double RC_skull_brain;
	double TC_skull_brain;
	double RC_skull_water;
	double TC_skull_water;
	float3 in_out_dist;
	float3 in_reflection_dist;
	double travel_length;
	double travel_length_reflection;
	cuDoubleComplex phase_refraction;
	cuDoubleComplex phase_reflection;
	double out_rc_sum = 0;
	double test_rc_sum = 0;
	double ARC;
	double Out_ARC;



	int idx = launch_index.x + launch_index.y;

		float3 ray_origin = origins[idx];
		float3 ray_direction = normalize(target - ray_origin);

		////////////////////////////////////////////
		//first ray:: transducer -> skull layer1
		///////////////////////////////////////////
		Ray ray(ray_origin, ray_direction, BEAM_RAY_TYPE, scene_epsilon);
		//rtPrintf("%d ray_origin == : [%f %f %f]\n", idx, ray_origin.x, ray_origin.y, ray_origin.x);
		PerRayData_beam first_prd;
		first_prd.isHit = false;
		first_prd.isDone = false;
		rtTrace(top_object, ray, first_prd);

		float3 out_skull_entervector;
		////////////////////////////////////////////
		//out_rc:: transducer -> skull layer1	
		///////////////////////////////////////////
		if (first_prd.isHit) {

			out_skull_entervector = target - ray_origin;
			angle_outskull = getIncidenceAngle(out_skull_entervector, first_prd.ffnormal);
			out_rc = reflection_coefficient(angle_outskull, random_density, random_speed, skull_density, skull_speed);
			RC_water_skull = out_rc;
			TC_water_skull = 1 - out_rc;
		}


		////////////////////////////////////////////
		//second ray::  skull layer1 -> skull layer2
		///////////////////////////////////////////
		PerRayData_beam second_prd;
		second_prd.isHit = true;


		////////////////////////////////////////////
		//R1::  skull layer1 -> skull layer2 refraction vector
		///////////////////////////////////////////
		float3 Refract1 = calVecRefraction(out_skull_entervector, first_prd.ffnormal, skull_speed, random_speed);
		float3 temp_hit_pos1 = first_prd.hit_pos + 0.1 *Refract1;
		float3 temp_normal1;

		if (!((Refract1.x == 0.0) && (Refract1.y == 0.0) && (Refract1.z == 0.0)))
		{

			////////////////////////////////////////////
			//temp_hit_pos:: to ignore artifacts between skull layers
			///////////////////////////////////////////

			while (second_prd.isHit) {
				Ray refraction1(temp_hit_pos1, Refract1, BEAM_RAY_TYPE, scene_epsilon);
				rtTrace(top_object, refraction1, second_prd);

				if (second_prd.isHit)
				{

					temp_hit_pos1 = second_prd.hit_pos;
					temp_normal1 = second_prd.ffnormal;
				}
			}
		}
		second_prd.isHit = !second_prd.isHit;

		////////////////////////////////////////////
		//in_rc:: skull layer1 -> skull layer2
		///////////////////////////////////////////
		if (second_prd.isHit)
		{
			angle_inskull = getIncidenceAngle(Refract1, temp_normal1);
			in_rc = reflection_coefficient(angle_inskull, skull_density, skull_speed, water_density, water_speed);
			RC_skull_brain = in_rc;
			TC_skull_brain = 1 - in_rc;
		}

		in_out_dist = first_prd.hit_pos - temp_hit_pos1;
		travel_length = magnitude_cu(in_out_dist);
		phase_refraction = _exp(make_cuDoubleComplex(0, k*travel_length));

		PerRayData_beam reflect_prd;
		reflect_prd.isHit = false;
		float3 R2 = calVecRefraction(Refract1, temp_normal1, water_speed, skull_speed);

		float3 temp_hit_pos2;
		float3 temp_normal2;
		if (!((R2.x == 0) && (R2.y == 0) && (R2.z == 0)))
		{

			////////////////////////////////////////////
			//reflect_prd:: skull layer2 -> skull layer1 reflect ray
			////////////////////////////////////////////
			float3 reflection_start = temp_hit_pos1 - 0.1 * Refract1;
			float3 in_reflection = calVecReflection(Refract1, temp_normal1);


			////////////////////////////////////////////
			//temp_prd:: keep rt until layer 1 (ignore artifacts between skull layers)
			///////////////////////////////////////////
			PerRayData_beam reflection_prd;
			reflection_prd.isHit = true;
			temp_hit_pos2 = reflection_start;

			while (reflection_prd.isHit) {
				Ray reflection(temp_hit_pos2, in_reflection, BEAM_RAY_TYPE, scene_epsilon);
				rtTrace(top_object, reflection, reflection_prd);
				if (reflection_prd.isHit)
				{
					temp_hit_pos2 = reflection_prd.hit_pos;
					temp_normal2 = reflection_prd.ffnormal;
				}
			}

			reflection_prd.isHit = !reflection_prd.isHit;
			if (reflection_prd.isHit) {
				float3 reflection_vector = (temp_hit_pos1 + 35 * in_reflection) - reflection_start;
				float3 reflection_vector_from_innerskull = calVecRefraction(reflection_vector, temp_normal2, random_speed, skull_speed);

				angle_reflection = getIncidenceAngle(reflection_vector, temp_normal2);
				reflect_rc = reflection_coefficient(angle_reflection, skull_density, skull_speed, random_density, random_speed);
				RC_skull_water = reflect_rc;
				TC_skull_water = 1 - reflect_rc;
			}
			in_reflection_dist = temp_hit_pos1 - temp_hit_pos2;
			travel_length_reflection = magnitude_cu(in_reflection_dist);
			phase_reflection = _exp(make_cuDoubleComplex(0, k*travel_length_reflection));

		}

		if (out_rc == 1) {
			test_rc = 1;
		}
		else {

			cuDoubleComplex result = cuCmul(phase_refraction, make_cuDoubleComplex(TC_water_skull, 0));
			result = cuCmul(result, make_cuDoubleComplex(RC_skull_brain, 0));
			result = cuCmul(result, phase_reflection);
			result = cuCmul(result, make_cuDoubleComplex(TC_skull_water, 0));
			test_rc = cuCabs(cuCadd(result, make_cuDoubleComplex(RC_water_skull, 0)));
			
		}

		//rtPrintf("%d ray_origin == : [%f %f %f]\n", idx, ray_origin.x, ray_origin.y, ray_origin.z);
		//rtPrintf("%d == first hit: [%f, %f, %f]\n", idx, first_prd.hit_pos.x, first_prd.hit_pos.y, first_prd.hit_pos.z);
		//rtPrintf("%d == norma1: [%f, %f, %f]\n", idx, first_prd.ffnormal.x, first_prd.ffnormal.y, first_prd.ffnormal.z);
		//rtPrintf("%d == second hit: [%f, %f, %f]\n", idx, temp_hit_pos1.x, temp_hit_pos1.y, temp_hit_pos1.z);
		//rtPrintf("%d == norma2: [%f, %f, %f]\n", idx, temp_normal1.x, temp_normal1.y, temp_normal1.z);
		//rtPrintf("%d == refr2: [%f, %f, %f]\n", idx, R2.x, R2.y, R2.z);
		//rtPrintf("%d == reflect hit: [%f, %f, %f]\n", idx, temp_hit_pos2.x, temp_hit_pos2.y, temp_hit_pos2.z);
		//rtPrintf("%d == normarefle: [%f, %f, %f]\n", idx, temp_normal2.x, temp_normal2.y, temp_normal2.z);
		//rtPrintf("%d ==  out_rc %f\n", idx, out_rc);
		//rtPrintf("%d == test_rc %f\n", idx, test_rc);


		out_rc_buffer[launch_index] = out_rc;
		test_rc_buffer[launch_index] = test_rc;
		out_intersection_point[launch_index] = first_prd.hit_pos;
		in_intersection_point[launch_index] = second_prd.hit_pos;
		reflec_in_skull2out[launch_index] = reflect_prd.hit_pos;
	
}


RT_PROGRAM void closest_hit_skull()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 hit_point = ray.origin + t_hit * ray.direction;

	prd_beam.ray_direction = ray.direction;
	prd_beam.ffnormal = ffnormal;
	prd_beam.hit_pos = hit_point;
	prd_beam.isHit = true;

}


RT_PROGRAM void any_hit()
{ 

	rtTerminateRay();
	
}

RT_PROGRAM void miss()
{
	
	prd_beam.isHit = false;
	
}
 
RT_PROGRAM void exception()
{
	rtPrintExceptionDetails();
}

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

#include "tutorial.h"
 //static __device__ __inline__ float fresnel(float cos_theta_i, float cos_theta_t, float eta);


//rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(PerRayData_beam, prd_beam, rtPayload, );

rtDeclareVariable(optix::Ray,  ray,          rtCurrentRay, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );

rtDeclareVariable(float, refraction_index, , );
//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        transducer_eye, , );
rtDeclareVariable(float3,		   transducer_U, , );
rtDeclareVariable(float3,		   transducer_V, , );
rtDeclareVariable(float3,		   transducer_W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;


using namespace optix;

RT_PROGRAM void ray_generation()
{
	size_t2 screen = output_buffer.size();
	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;

	float3 ray_origin = transducer_eye;
	//float3 ray_origin2 = make_float3(-5.0f, 60.0f, -16.0f);
	//float3 ray_direction = normalize(d.x*transducer_U + d.y*transducer_V + transducer_W);
	float3 ray_direction = normalize(make_float3(0.0f, 4.0f, 0.0f) - ray_origin);
	Ray ray(ray_origin, ray_direction, BEAM_RAY_TYPE, scene_epsilon);


	PerRayData_beam first_prd;
	first_prd.depth = 0;
	rtTrace(top_object, ray, first_prd);
	rtPrintf("1: [%f, %f, %f]\n", first_prd.hit_pos.x, first_prd.hit_pos.y, first_prd.hit_pos.z);

	PerRayData_beam second_prd;
	float3 R1;
	rtPrintf("/////////////// [%f, %f, %f]\n", first_prd.ffnormal.x, first_prd.ffnormal.y, first_prd.ffnormal.z);
	if (refract(R1, normalize(first_prd.ray_direction), first_prd.ffnormal, refraction_index))
	{ //1.4f
		Ray refraction1(first_prd.hit_pos, R1, BEAM_RAY_TYPE, scene_epsilon);
		rtTrace(top_object, refraction1, second_prd);
	}
	//rtprintf("2:  [%f, %f, %f]\n", r1.x, r1.y, r1.z);
	rtPrintf("\n\n 2:  [%f, %f, %f]\n", second_prd.hit_pos.x, second_prd.hit_pos.y, second_prd.hit_pos.z);


	PerRayData_beam third_prd;
	float3 R2;
	if (refract(R2, normalize(second_prd.ray_direction), second_prd.ffnormal, refraction_index)) {
		Ray refraction2(second_prd.hit_pos, R2, BEAM_RAY_TYPE, scene_epsilon);
		rtTrace(top_object, refraction2, third_prd);
	}
	//rtPrintf("2:  [%f, %f, %f]\n", R2.x, R2.y, R2.z);
	rtPrintf("\n\n 3: [%f, %f, %f]\n", third_prd.hit_pos.x, third_prd.hit_pos.y, third_prd.hit_pos.z);
}


RT_PROGRAM void closest_hit_beam()
{
//  prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 hit_point = ray.origin + t_hit * ray.direction;


	prd_beam.ray_direction = ray.direction;
	prd_beam.ffnormal = ffnormal;
	prd_beam.hit_pos = hit_point;
	//prd_beam.depth = prd_beam.depth + 1;
}

RT_PROGRAM void any_hit()
{
	rtTerminateRay();
}
 
//RT_PROGRAM void miss()
//{
//	rtPrintf("%d: miss\n", prd_beam.depth);
//}



//
// Set pixel to solid color upon failur
//
RT_PROGRAM void exception()
{

	rtPrintExceptionDetails();
 //output_buffer[launch_index] = make_color( bad_color );
}

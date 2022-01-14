#include <optix.h>
#include <optix_math.h>
#include "common.h"
#define UNIFIED_MATH_CUDA_H
#include "vector_functions.h"

using namespace optix;

struct PerRayData_beam
{
	float3 result;
	float  importance;
	int depth;
	float3 hit_pos;
	float3 ray_direction;
	float3 ffnormal;
	bool isHit;
	bool isDone;
	bool temp;
	bool isSecond;
	int num;
};


static __device__ __inline__ float3 exp(const float3& x)
{
	return make_float3(exp(x.x), exp(x.y), exp(x.z));
}


static __device__ __inline__ float step(float min, float value)
{
	return value < min ? 0 : 1;
}

static __device__ __inline__ float3 mix(float3 a, float3 b, float x)
{
	return a * (1 - x) + b * x;
}

static __device__ __inline__ float3 schlick(float nDi, const float3& rgb)
{
	float r = fresnel_schlick(nDi, 5, rgb.x, 1);
	float g = fresnel_schlick(nDi, 5, rgb.y, 1);
	float b = fresnel_schlick(nDi, 5, rgb.z, 1);
	return make_float3(r, g, b);
}

float magnitude(float3 vector) {

	return sqrt(pow(vector.x, 2) + pow(vector.y, 2) + pow(vector.z, 2));
}



double l2n(float3 data)
{
	return sqrt(data.x * data.x + data.y * data.y + data.z * data.z);
}

double degrees(double a)
{
	return  a * 180.0 / 3.141592654f;
	
}

double radians(double a)
{
	return  a * 3.141592654f / 180.0;
}



double reflection_coefficient(double angle, float up_density, float up_speed, float down_density, float down_speed)
{
	double b, c, test, reflection;

	double theta = radians(angle);

	double m = down_density / up_density;
	double n = up_speed / down_speed;

	// test for critical angle
	test = (1 - ((sinf(theta) * sinf(theta)) / (n * n)));
	test = (test < 0) ? 0.0f : test;


	b = m * cosf(theta);
	c = n * sqrt(test);
	
	reflection = abs((b - c) / (b + c));
	reflection = (reflection > 1) ? 1.0f : reflection;
	
	return reflection;
}


float magnitude_cu(float3 vector) {

	return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z *vector.z);
}

double getIncidenceAngle(float3 rayDir, float3 ffnormal)
{
	float negNdotV = dot(rayDir, ffnormal);
	float angle = acos(negNdotV / magnitude_cu(rayDir) * magnitude_cu(ffnormal));
	
	//180 - degrees(
	return 180 - degrees(angle);
}

float3 calVecReflection(float3 enter_vector,float3 normal_skull) {

	//this def calculate angle between two vectors
	//return refraction vector and tilted angle
	
	float3	S1 = enter_vector / magnitude_cu(enter_vector);
	float3	N = normal_skull / magnitude_cu(normal_skull);


	float3	reflection_vector = enter_vector - 2 * dot(S1, N)*N;


	return reflection_vector;
}


float3 calVecRefraction(float3 enter_vector, float3  normal_skull, float  out_m, float  in_m) {

	//# this def calculate angle between two vectors
	//return refraction vector and tilted angle

	float3	S1 = normalize(enter_vector);
	float3	N = normalize(normal_skull);
	float3 S2;
	float	n = out_m / in_m;

	float crossP = dot(N, S1);
	//float crossN = np.cross(-N, S1);
	float T = (1 - (n * n) * (1 - dot(S1, N) * dot(S1, N)));


	if (T > 0) {
		//S2 = (A - N * np.sqrt(B))
		S2 = n * (S1 + N * dot(S1, -N)) - N * sqrt(1 - (n * n) * (1 - dot(S1, N) * dot(S1, N)));
	}
	else
		S2 = make_float3(0, 0, 0);

	return S2;

}





#pragma once
#include "raytracing.h"
#include <iostream>
//#include <complex> 
#include <vector>
#include <functional>
#include <algorithm>
#include <vtkSTLReader.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkDelaunay3D.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkTransform.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPointLocator.h>
#include <vtkPlane.h>
#include <vtkCylinderSource.h>
#include <vtkClipPolyData.h>
#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>
#include <vtkSphereSource.h>
#include <vtkOBJExporter.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkPolyDataNormals.h>
#include "vtkOBJWriter.h"
#include <vtkSTLWriter.h>
//#include <vtkOBJWriter.h>

typedef struct _double3
{
	double x;
	double y;
	double z;
} double3;

template<typename T>
std::vector<float> arange(T start, T stop, T step = 1) {
	std::vector<float> values;
	for (float value = start; value < stop; value += step)
		values.push_back(value);
	return values;
}

template<typename T1, typename T2>
std::vector<float> linspace(T1 start_in, T2 end_in, const int num_in)
{
	std::vector<float> linspaced;

	double start = static_cast<float>(start_in);
	double end = static_cast<float>(end_in);
	double num = static_cast<float>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for (int i = 0; i < num - 1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
							  // are exactly the same as the input
	return linspaced;
}


float getMinDistance(std::vector<float3> coordi_list, float3 coordi) {

	////0이하면 추가 안하는지 확인 필요
	std::vector<float> distance;
	for (auto &coordiL: coordi_list) // access by reference to avoid copying
	{
		float dis = sqrt(pow(coordiL.x - coordi.x, 2) + pow(coordiL.y - coordi.y, 2) + pow(coordiL.z - coordi.z, 2));
		if (dis > 0)
			distance.push_back(dis);
	}

	// min_element return address not value
	return 	*std::min_element(distance.begin(), distance.end());
}


vtkSmartPointer<vtkPolyData> read_skull(std::string filename) {
	
	vtkSmartPointer<vtkSTLReader> readerStl = vtkSmartPointer<vtkSTLReader>::New();
	//readerStl->SetFileName(filename.c_str());
	readerStl->SetFileName((std::string(sutil::samplesDir()) + "/data/skull-smooth2.stl").c_str());
	
	readerStl->Update();

	vtkSmartPointer<vtkPolyData> reader = readerStl->GetOutput();

	return reader;
}





vtkSmartPointer<vtkPolyData> cut_skull(vtkSmartPointer<vtkPolyData> skull,float3 Target,float3 centerline_vector)
{

	float3 cutting_center = Target + (centerline_vector * -16);
	
	vtkSmartPointer<vtkPlane> plane = vtkSmartPointer<vtkPlane>::New();
	plane->SetOrigin(cutting_center.x , cutting_center.y, cutting_center.z);
	plane->SetNormal(centerline_vector.x, centerline_vector.y, centerline_vector.z);

	//vtkSmartPointer<vtkCylinderSource> cylinderSource =	vtkSmartPointer<vtkCylinderSource>::New();
	//cylinderSource->SetCenter(Target.x, Target.y, Target.z);
	//cylinderSource->SetRadius(50.0);

	vtkSmartPointer<vtkClipPolyData> clipper = vtkSmartPointer<vtkClipPolyData>::New();
	clipper->SetInputData(skull);
	clipper->SetClipFunction(plane);
	clipper->SetValue(0);
	clipper->Update();

	vtkSmartPointer<vtkPolyData> skull_cut = clipper->GetOutput();
	
	return skull_cut;
}

std::tuple < vtkSmartPointer<vtkPolyData>, float, float> make_transducer(vtkSmartPointer<vtkPolyData> spherePoly, int ROC, int width, float focal_length, float3 range_vector, float3 Target) {

	float3 center_vector = make_float3(1, 0, 0);
	float3 unit_vector = normalize(range_vector);
	float3 xy_unit_vector = make_float3(unit_vector.x, unit_vector.y, 0);


	float xy_angle;
	float z_angle;

	if ((xy_unit_vector.x == 0.0 )&& (xy_unit_vector.y == 0.0) && (xy_unit_vector.z == 0.0))
	{
		xy_angle = 0.0f;
		z_angle = 90.0f;
	}
	else
	{
		xy_angle = degrees(acos(
			dot(center_vector, xy_unit_vector) /
			(magnitude(center_vector) * magnitude(xy_unit_vector))
		));
		z_angle = degrees(acos(
			dot(xy_unit_vector, unit_vector) /
			(magnitude(xy_unit_vector) * magnitude(unit_vector))
		));
	}

	if (unit_vector.z < 0)
		z_angle = -1 * z_angle;
	if (unit_vector.y < 0)
		xy_angle = -1 * xy_angle;

	///transform(rotation)
	float gap = focal_length - ROC;
	float3 GAP = Target + range_vector * gap;

	//////transform & rotation
	vtkSmartPointer<vtkTransform> translation = vtkSmartPointer<vtkTransform>::New();
	translation->Translate(GAP.x, GAP.y, GAP.z);
	translation->RotateWXYZ(90, 0, 1, 0);
	translation->RotateWXYZ(-xy_angle, 1, 0, 0);
	translation->RotateWXYZ(-z_angle, 0, 1, 0);

	vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	transformFilter->SetTransform(translation);
	transformFilter->SetInputData(spherePoly);
	transformFilter->Update();

	return std::make_tuple(transformFilter->GetOutput(), xy_angle, z_angle);
}


vtkSmartPointer<vtkPolyData> cut_skull_loop(vtkSmartPointer <vtkPolyData> skull,float3 Target,float3 centerline_vector){
	
	vtkSmartPointer<vtkPlane> plane = vtkSmartPointer<vtkPlane>::New();
	plane->SetNormal(centerline_vector.x, centerline_vector.y, centerline_vector.z);

	vtkSmartPointer<vtkCylinderSource> cylinderSource = vtkSmartPointer<vtkCylinderSource>::New();
	cylinderSource->SetCenter(Target.x, Target.y, Target.z);
	cylinderSource->SetRadius(50.0);


	vtkSmartPointer<vtkClipPolyData> clipper = vtkSmartPointer<vtkClipPolyData>::New();
	clipper->SetInputData(skull);
	clipper->SetClipFunction(plane);
	clipper->SetValue(0);
	clipper->Update();

	return clipper->GetOutput();
}

vtkSmartPointer<vtkSphereSource> addPoint(float3 p, float radius) {

	vtkSmartPointer<vtkSphereSource> point = vtkSmartPointer<vtkSphereSource>::New();
	point->SetCenter(p.x, p.y, p.z);
	point->SetRadius(radius);
	point->SetPhiResolution(100);
	point->SetThetaResolution(100);

	return point;
}


float3 make_centerline_target(vtkSmartPointer<vtkPolyData> skull, float3 target, float centerline_length) {

	vtkSmartPointer<vtkPointLocator> pointLocator = vtkSmartPointer<vtkPointLocator>::New();
	pointLocator->SetDataSet(skull);
	pointLocator->BuildLocator();

	double p[3];
	skull->GetPoint(pointLocator->FindClosestPoint(target.x, target.y, target.z), p);

	float3	vector = make_float3(p[0] - target.x, p[1] - target.y, p[2] - target.z);
	//chexk
	//float3 centerline_vector = vector / magnitude(vector);
	float3 centerline_vector = normalize(vector);
	//	centerline_vector = vector / np.linalg.norm(vector)
	//	centerline_target = n2l(l2n(target) + centerline_length * centerline_vector)

	return centerline_vector;
}



vtkSmartPointer<vtkPolyData> make_evencirle(int num_pts, int ROC, int width, float focal_length) {
	//make transducer function with evely distributed spots
	float h_wid, p_height, height_from_center, height, rate;
	std::vector<float> indices_theta, theta, indices_phi, phi;
	std::vector <float3>  coordi_list;
	std::vector <float>  x, y, z;

	h_wid = width / (float)2;
	p_height = pow(ROC, 2) - pow(h_wid, 2);
	height_from_center = sqrt(p_height);
	height = ROC - height_from_center;  // transducer's height
	rate = height / (ROC * 2);  // ratio height / ROC * 2


	indices_theta = arange(0, num_pts, 1);
	indices_phi = linspace(0, num_pts * rate, num_pts);// define transdcuer's height as ratio

	for (float &data : indices_phi) {
		float temp = acos(1 - 2 * (data / num_pts));
		phi.push_back(temp);
	}
	for (float &data : indices_theta) {
		float temp = M_PI * (1 + pow(5, 0.5)) * data;
		theta.push_back(temp);
	}

	for (int i = 0; i < num_pts; i++) {
		float3 coordi;

		coordi.x = cosf(theta.at(i)) * sinf(phi.at(i))*ROC;
		coordi.y = sinf(theta.at(i)) * sinf(phi.at(i))*ROC;
		coordi.z = cosf(phi.at(i))*ROC;
		coordi_list.push_back(coordi);
	}

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	std::vector<float> dis_min;
	double dis_sum = 0;
	for (auto &coordi : coordi_list) // access by reference to avoid copying
	{
		double min = getMinDistance(coordi_list, coordi);
		dis_min.push_back(min);
		dis_sum += min;
		points->InsertNextPoint(coordi.x, coordi.y, coordi.z); // x, y ,z
	}

	float dis_average = dis_sum / dis_min.size();

	// Create a polydata object
	vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
	poly->SetPoints(points);

	vtkSmartPointer<vtkDelaunay3D> delaunay3D = vtkSmartPointer<vtkDelaunay3D>::New();
	delaunay3D->SetInputData(poly);

	vtkSmartPointer<vtkDataSetSurfaceFilter> surfaceFilter = vtkSmartPointer<vtkDataSetSurfaceFilter>::New();
	surfaceFilter->SetInputConnection(delaunay3D->GetOutputPort());
	surfaceFilter->Update();

	vtkSmartPointer<vtkPolyData> spherePoly = surfaceFilter->GetOutput();

	return spherePoly;
}


std::tuple< vtkSmartPointer<vtkPolyData>, float, std::vector<float>>  make_analysis_rage(const int num_pts, float radius, int range_angle, float3 centerline_vector, float3 target)
{
	float h_wid, p_height, height_from_center, height, rate;
	std::vector<float> indices_theta, theta, indices_phi, phi;


	std::vector <float3>  coordi_list;
	std::vector <float>  x, y, z;

	h_wid = radius * sinf(radians(range_angle));
	p_height = radius * radius - h_wid * h_wid;
	height_from_center = sqrt(p_height);
	height = radius - height_from_center;

	rate = height / (radius * 2);

	indices_theta = arange(0, num_pts, 1);
	indices_phi = linspace(0, num_pts * rate, num_pts);


	for (float &data : indices_phi) {
		float temp = acos(1 - 2 * (data / num_pts));
		phi.push_back(temp);
	}
	for (float &data : indices_theta) {
		float temp = M_PI * (1 + pow(5, 0.5)) * data;
		theta.push_back(temp);
	}


	for (int i = 0; i < num_pts; i++) {
		float3 coordi;

		coordi.x = cosf(theta.at(i)) * sinf(phi.at(i))*radius;
		coordi.y = sinf(theta.at(i)) * sinf(phi.at(i))*radius;
		coordi.z = cosf(phi.at(i))*radius;
		coordi_list.push_back(coordi);
	}


	///////////////////////////////vtk  to get cutpoly
	// Create the geometry of a point (the coordinate)

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	std::vector<float> dis_min;
	double dis_sum = 0;
	for (auto &coordi : coordi_list) // access by reference to avoid copying
	{
		double min = getMinDistance(coordi_list, coordi);
		dis_min.push_back(min);
		dis_sum += min;
		points->InsertNextPoint(coordi.x, coordi.y, coordi.z); // x, y ,z
	}

	float dis_average = dis_sum / dis_min.size();

	// Create a polydata object
	vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
	poly->SetPoints(points);

	vtkSmartPointer<vtkDelaunay3D> delaunay3D = vtkSmartPointer<vtkDelaunay3D>::New();
	delaunay3D->SetInputData(poly);

	vtkSmartPointer<vtkDataSetSurfaceFilter> surfaceFilter = vtkSmartPointer<vtkDataSetSurfaceFilter>::New();
	surfaceFilter->SetInputConnection(delaunay3D->GetOutputPort());
	surfaceFilter->Update();

	vtkSmartPointer<vtkPolyData> spherePoly = surfaceFilter->GetOutput();


	float3 center_vector = make_float3(1.0, 0.0, 0.0);

	float norm_v = magnitude(centerline_vector);
	float3 unit_vector = make_float3(centerline_vector.x / norm_v, centerline_vector.y / norm_v, centerline_vector.z / norm_v);

	float3 xy_unit_vector = make_float3(unit_vector.x, unit_vector.y, 0);

	float xy_angle;
	float z_angle;

	if ((xy_unit_vector.x == 0.0) && (xy_unit_vector.y) == 0.0 && (xy_unit_vector.z == 0.0))
	{
		xy_angle = 0.0f;
		z_angle = 90.0f;
	}
	else
	{
		xy_angle = degrees(acos(
			dot(center_vector, xy_unit_vector) /
			(magnitude(center_vector) * magnitude(xy_unit_vector))
		));
		z_angle = degrees(acos(
			dot(xy_unit_vector, unit_vector) /
			(magnitude(xy_unit_vector) * magnitude(unit_vector))
		));
	}

	if (unit_vector.z < 0)
		z_angle = -1 * z_angle;
	if (unit_vector.y < 0)
		xy_angle = -1 * xy_angle;

	//////transform & rotation

	vtkSmartPointer<vtkTransform> translation = vtkSmartPointer<vtkTransform>::New();
	translation->Translate(target.x, target.y, target.z);
	translation->RotateWXYZ(90, 0, 1, 0);
	translation->RotateWXYZ(-xy_angle, 1, 0, 0);
	translation->RotateWXYZ(-z_angle, 0, 1, 0);

	vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	transformFilter->SetInputData(spherePoly);
	transformFilter->SetTransform(translation);
	transformFilter->Update();

	vtkSmartPointer<vtkPolyData> cutpoly = transformFilter->GetOutput();

	return  std::make_tuple(cutpoly, dis_average, dis_min);
}

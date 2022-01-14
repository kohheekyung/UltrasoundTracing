
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>
#include <OptiXMesh.h>
#include "calculate.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <array>
#include <numeric>
#include <thread>
#include <fstream>
#include <algorithm>
using namespace optix;
const char* const SAMPLE_NAME = "optixPathFinder";

optix::Context        context;
int      width = 1250;
int       height = 1250;
bool           use_pbo = true;
bool           use_tri_api = true;
bool           ignore_mats = false;


const char *t_ptx = sutil::getPtxString(SAMPLE_NAME, "raytracing.cu");
std::map<std::string,  RTsize> buffer_map;

const int transducerNum = 1250;
const int number_of_beamlines = 100;
int centerline_length = 65;

//spec
float focal_length = 55.22;
int ROC = 71;
int t_width = 65;
float length_transducer2target = 55.22;
int range_angle = 45;

//vtk configure
vtkSmartPointer<vtkPolyData> spherePoly;
vtkSmartPointer<vtkPolyData> a_range;
vtkSmartPointer<vtkSphereSource> focus;
vtkSmartPointer<vtkPolyData> first_cutskull;

float3 Target = make_float3(22.7292 , -27.1715 , 56.572); //SMA
//------------------------------------------------------------------------------
// 
// Forward decls 
//
//------------------------------------------------------------------------------
Buffer getOutputBuffer();
Buffer getOutputBuffer_t(std::string s);
void createContext( );
void loadMesh( const std::string& filename );
void destroybuffers();

std::tuple< double, double> getData(int position_num);
void setProperties();

void vtkconfigure(const std::string& filedate);
std::tuple < float ,
	std::vector<double>,
	std::vector<double>,
	std::vector<double>,
	std::vector<double>,
	std::vector<float3> > do_work(int begin, int end);

void writefile(const std::string& filedate,
	int optimal_transducer_num,
	std::vector<double> ARC_list,
	std::vector<double> out_ARC_list,
	std::vector<double> xy_angle,
	std::vector<double> z_angle,
	std::vector<float3> transducer_location);
//
//float3 make_centerline_target(vtkSmartPointer<vtkPolyData> skull, float3 target, float centerline_length);
//vtkSmartPointer<vtkPolyData> make_evencirle(int num_pts = 1000, int ROC = 71, int width = 65, float focal_length = 55.22);
//std::tuple< vtkSmartPointer<vtkPolyData>, float, std::vector<float>>  make_analysis_rage(const int num_pts, float radius, int range_angle, float3 centerline_vector, float3 target);
//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void createContext()
{
	// Set up context
	context = Context::create();
	context->setRayTypeCount(3);
	context->setEntryPointCount(1);
	context->setStackSize(4640);
	context->setMaxTraceDepth(1);
	context->setPrintEnabled(1);
	context->setPrintBufferSize(4096);

	context["scene_epsilon"]->setFloat(1.e-4f);

	//////////////////////////////////////////////////////////
	Buffer out_rc_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, 1, number_of_beamlines);
	out_rc_buffer->setElementSize(sizeof(double));
	context["out_rc_buffer"]->set(out_rc_buffer);
	buffer_map.insert(std::make_pair("out_rc_buffer", sizeof(double)));

	Buffer test_rc_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, 1, number_of_beamlines);
	test_rc_buffer->setElementSize(sizeof(double));
	context["test_rc_buffer"]->set(test_rc_buffer);
	buffer_map.insert(std::make_pair("test_rc_buffer", sizeof(double)));

	Buffer out_intersection_point = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 1, number_of_beamlines);
	context["out_intersection_point"]->set(out_intersection_point);
	buffer_map.insert(std::make_pair("out_intersection_point", sizeof(float3)));

	Buffer  in_intersection_point = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 1, number_of_beamlines);
	context["in_intersection_point"]->set(in_intersection_point);
	buffer_map.insert(std::make_pair("in_intersection_point", sizeof(float3)));

	Buffer reflec_in_skull2out = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 1, number_of_beamlines);
	context["reflec_in_skull2out"]->set(reflec_in_skull2out);
	buffer_map.insert(std::make_pair("reflec_in_skull2out", sizeof(float3)));


	// Ray generation program raytracing.cu
	Program ray_gen_program = context->createProgramFromPTXString(t_ptx, "ray_generation");
	context->setRayGenerationProgram(0, ray_gen_program);

	// Exception program  raytracing.cu
	Program exception_program = context->createProgramFromPTXString(t_ptx, "exception");
	context->setExceptionProgram(0, exception_program);

	// Miss program raytracing.cu set ray_type 2 not entrypoint 0
	context->setMissProgram(2, context->createProgramFromPTXString(t_ptx, "miss"));

}

Buffer getOutputBuffer_t(std::string s)
{
	return context[s]->getBuffer();
}


void destroybuffers() {

	for (auto it = buffer_map.begin(); it != buffer_map.end(); it++) {

		Buffer buffer = getOutputBuffer_t(it->first);
		buffer->destroy();
	}
}
void setProperties()
{
	context["water_density"]->setFloat(998.2);
	context["water_speed"]->setFloat (1482.0);
	context["skull_density"]->setFloat(1732.0);
	context["skull_speed"]->setFloat(2850.0);
	context["random_density"]->setFloat(998.2); //water_density
	context["random_speed"]->setFloat(1482.0); //water_speed transducer to skull properties
}

void loadMesh( const std::string& filedate )
{

	const char *t_ptx = sutil::getPtxString("optixPathFinder", "raytracing.cu");
	optix::Material t_matl = context->createMaterial();
	Program t_ch = context->createProgramFromPTXString(t_ptx, "closest_hit_skull");
	Program t_ah = context->createProgramFromPTXString(t_ptx, "any_hit");
	t_matl->setClosestHitProgram(2, t_ch);
	t_matl->setAnyHitProgram(2, t_ah);

	

	std::string mesh_file = "CUT_skull" + filedate + ".obj";
    OptiXMesh mesh;
	//mesh.skull = true;
    mesh.context = context;
	mesh.ignore_mats = false;
    mesh.use_tri_api = use_tri_api; 
	mesh.material = t_matl;
    loadMesh(mesh_file, mesh );

    
	//aabb.set(mesh2.bbox_min, mesh2.bbox_max);
    GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance); 
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    context[ "top_object"   ]->set( geometry_group ); 
    context[ "top_shadower" ]->set( geometry_group ); 
	context["target"]-> setFloat(Target);

}

  


std::tuple<  double, double> getData(int position_num)
{
	double ARC = 0;
	double out_ARC = 0;
	double out_rc_sum = 0;
	long double test_rc_sum = 0;
	int out_rc_idx = 0;
	int test_rc_idx = 0;
	int out_intersection_idx = 0;
	int in_intersection_idx = 0;
	int reflec_in_idx = 0;
	/*std::array<double, number_of_beamlines> out_rc_list = { 0 };
	std::array<double, number_of_beamlines> test_rc_list = { 0 };
	std::array<float3, number_of_beamlines> out_intersection_point = { make_float3(0,0,0) };
	std::array<float3, number_of_beamlines> in_intersection_point = { make_float3(0,0,0) };
	std::array<float3, number_of_beamlines> reflec_in_skull2out = { make_float3(0,0,0) };*/

	std::vector<double> out_rc_list; out_rc_list.resize(number_of_beamlines);
	std::vector<double> test_rc_list; test_rc_list.resize(number_of_beamlines);
	std::vector<float3> out_intersection_point; out_intersection_point.resize(number_of_beamlines);
	std::vector<float3> in_intersection_point; in_intersection_point.resize(number_of_beamlines);
	std::vector<float3> reflec_in_skull2out; reflec_in_skull2out.resize(number_of_beamlines);

	for (auto it = buffer_map.begin(); it != buffer_map.end(); it++) {
		Buffer buffer = getOutputBuffer_t(it->first);

		if (it->second == sizeof(float3)) {
			float3* data = static_cast<float3*>(buffer->map());
			if (!data) {
				std::cerr << "Can't map output buffer\n";
				exit(2);
			}

			RTsize buffer_width, buffer_height;
			buffer->getSize(buffer_width, buffer_height);

			for (RTsize y = 0; y < buffer_height; ++y) {

				float3* row = data + ((buffer_height - y - 1)*buffer_width);

				for (RTsize x = 0; x < buffer_width; ++x) {
					float3 data = row[x];


					//out << it->first << ":	\t" << data << "\n";;

					if ((it->first).compare("out_intersection_point") == 0) {
						out_intersection_point[out_intersection_idx++] = data;
					}
					else if ((it->first).compare("in_intersection_point") == 0) {
						in_intersection_point[in_intersection_idx++] = data;
					}
					else if ((it->first).compare("reflec_in_skull2out") == 0) {
						reflec_in_skull2out[reflec_in_idx++] = data;
					}





				}
			}
			//std::cout << out.str();
			buffer->unmap();
		}
		//double data
		else if (it->second == sizeof(double)) {

			double* data = static_cast<double*>(buffer->map());
			if (!data) {
				std::cerr << "Can't map output buffer\n";
				exit(2);
			}

			RTsize buffer_width, buffer_height;
			buffer->getSize(buffer_width, buffer_height);

			for (RTsize y = 0; y < buffer_height; ++y) {

				double* row = data + ((buffer_height - y - 1)*buffer_width);

				for (RTsize x = 0; x < buffer_width; ++x) {
					double data = row[x];
					
					if ((it->first).compare("out_rc_buffer") == 0) {
						out_rc_list[out_rc_idx++] = data;
						out_rc_sum += data;
					}
					else if ((it->first).compare("test_rc_buffer") == 0) {
						//out << it->first << ":	\t" << data << "\n";
					/*	if (isnan(data))
							data = 0;*/
						test_rc_list[test_rc_idx++] = data;
						test_rc_sum += data;
					}
				}
			}
			//std::cout << out.str();
			buffer->unmap();
		}

	}
	out_rc_idx = 0;
	test_rc_idx = 0;
	out_intersection_idx = 0;
	in_intersection_idx = 0;
	reflec_in_idx = 0;

	ARC = test_rc_sum / number_of_beamlines;
	out_ARC = out_rc_sum / number_of_beamlines;
	
	if (isnan(ARC))
		ARC = 999999;
	if (isnan(out_ARC))
		out_ARC = 999999;

	return  std::make_tuple(ARC, out_ARC );
}




//std::array<int,1250> sort_indexes(const std::array < double, 1250> &v) {
//
//	// initialize original index locations
//	std::array<int, 1250> idx;
//	iota(idx.begin(), idx.end(), 0);
//
//	// sort indexes based on comparing values in v
//	sort(idx.begin(), idx.end(),
//		[&v](int i1, int i2) {return v[i1] < v[i2]; });
//
//	return idx;
//}

std::vector<int> sort_indexes(const std::vector<double> &v) {

	// initialize original index locations
	std::vector<int> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](int i1, int i2) {return v[i1] < v[i2]; });

	return idx;
}

void writefile(const std::string& filedate, int optimal_transducer_num, std::vector<double> ARC_list, std::vector<double> out_ARC_list, std::vector<double> xy_angle,  std::vector<double> z_angle, std::vector<float3> transducer_location) {

	std::string ARCfilePath = filedate + "_ARC_SMA_result.txt";
	std::string out_ARCfilePath = filedate + "_outARC_SMA_result.txt";


	//std::sort(ARC_list.begin(), ARC_list.end());
	//std::sort(out_ARC_list.begin(), out_ARC_list.end());
	std::cout << "Target :: SMA [22.7292, -27.1715, 56.572]" << endl;
	std::cout << "Minimum reflection coefficient is ::      " << ARC_list[optimal_transducer_num] << "\n";
	std::cout << "(from ARC result) The optimal position of transducer location is  ::      " << transducer_location[optimal_transducer_num] << "\n";

	std::vector<int> idx = sort_indexes(ARC_list);
	//cout << idx.size() << endl;
	//cout << ARC_list.size() << endl;
	//cout << xy_angle.size() << endl;
	// write File
	std::ofstream writeFile(ARCfilePath.data());
	if (writeFile.is_open()) {
		writeFile << "ARC\t\tnum\t\txy_angle\t\tz_angle\t\ttransducer_location\n";
		for (int i = 0; i < idx.size(); i++) {
			writeFile << ARC_list[idx[i]] << "\t\t" ;
			writeFile << idx[i] << "\t\t";
			writeFile << xy_angle[idx[i]] << "\t\t";
			writeFile << z_angle[idx[i]] << "\t\t";
			writeFile << transducer_location[idx[i]] << "\n";
		}
		writeFile.close();
	}

	std::vector<int> idx2 = sort_indexes(out_ARC_list);
	std::ofstream writeFile2(out_ARCfilePath.data());
	if (writeFile2.is_open()) {
		writeFile2 << "out_ARC\t\tnum\t\tz_angle\t\ttransducer_location\n";
		for (int i = 0; i < idx2.size(); i++) {
			writeFile2 << out_ARC_list[idx2[i]] << "\t\t";
			writeFile2 << idx2[i] << "\t\t";
			writeFile2 << xy_angle[idx2[i]] << "\t\t";
			writeFile2 << z_angle[idx2[i]] << "\t\t";
			writeFile2 << transducer_location[idx2[i]] << "\n";
		}
		writeFile2.close();
	}
}

void vtkconfigure(const std::string& filedate) {

	std::string skull_file_name = "skull-smooth2.stl";
	vtkSmartPointer<vtkPolyData> skull = read_skull(skull_file_name);
	float3 centerline_vector = make_centerline_target(skull, Target, centerline_length);
	spherePoly = make_evencirle(number_of_beamlines, ROC, t_width, focal_length);

	first_cutskull = cut_skull(skull, Target, centerline_vector);
	focus = addPoint(Target, 4);

	float transducer_mesh_mean;
	std::vector<float> transducer_mesh_dis;

	std::tie(a_range, transducer_mesh_mean, transducer_mesh_dis) = make_analysis_rage(transducerNum, length_transducer2target + 0, range_angle, centerline_vector, Target);
	std::string outputFilename = "CUT_skull" + filedate + ".obj";

	vtkSmartPointer<vtkOBJWriter> cutskull_writer = vtkSmartPointer<vtkOBJWriter>::New();
	cutskull_writer->SetInputData(first_cutskull);
	cutskull_writer->SetFileName(outputFilename.c_str());
	cutskull_writer->Update();

	loadMesh(filedate);
}

std::tuple<float, 
	std::vector<double>, 
	std::vector<double>, 
	std::vector<double> , 
	std::vector<double>,
	std::vector<float3> >  do_work(int begin, int end) {
	
	int ARC_min_idx = 0;
	int out_ARC_min_idx = 0;
	int optimal_transducer_num;

	std::vector<double> ARC_list;
	std::vector<double> out_ARC_list;
	std::vector<double> xy_angle;
	std::vector<double> z_angle;
	std::vector<float3> transducer_location;
	ARC_list.resize(transducerNum);
	out_ARC_list.resize(transducerNum);
	xy_angle.resize(transducerNum);
	z_angle.resize(transducerNum);
	transducer_location.resize(transducerNum);
	ARC_list[0] = 999999;
	out_ARC_list[0] = 999999;

	float3 top_point;
	for (int i = begin; i < end; i++) {
		top_point.x = (a_range->GetPoint(i))[0];
		top_point.y = (a_range->GetPoint(i))[1];
		top_point.z = (a_range->GetPoint(i))[2];
		
		transducer_location[i] = top_point;

		float3 vector = top_point - Target;
		float3 dir_vector = normalize(vector);

		vtkSmartPointer<vtkPolyData> transducer;
		std::tie(transducer, xy_angle[i], z_angle[i])
			= make_transducer(spherePoly, ROC, t_width, length_transducer2target + 0, dir_vector, Target);
		
		//float3 origin_p[100] ;
		std::vector<float3> origin_p;
		origin_p.resize(transducer->GetNumberOfPoints());
		for (int j = 0; j < transducer->GetNumberOfPoints(); j++) {
			origin_p[j] = make_float3(transducer->GetPoint(j)[0], transducer->GetPoint(j)[1], transducer->GetPoint(j)[2]);
		}

		Buffer point_buffer = context->createBuffer(RT_BUFFER_INPUT);
		point_buffer->setFormat(RT_FORMAT_USER);
		point_buffer->setElementSize(sizeof(float3));
		point_buffer->setSize(origin_p.size());
		memcpy(point_buffer->map(), &(origin_p[0]), sizeof(float3) * origin_p.size());
		point_buffer->unmap();
	
		context["origins"]->set(point_buffer);
		context->validate();
		context->launch(0, 1, number_of_beamlines);
		
		std::tie( ARC_list[i], out_ARC_list[i]) = getData(i);
	
		if (ARC_list[i] < ARC_list[ARC_min_idx]) {
			ARC_min_idx = i;
			optimal_transducer_num = i;
		}
		if (out_ARC_list[i] < ARC_list[out_ARC_min_idx])
			out_ARC_min_idx = i;
		
		point_buffer->destroy();
	}
	context->destroy();

	return  std::make_tuple(optimal_transducer_num, ARC_list, out_ARC_list, xy_angle, z_angle ,transducer_location );
}

int main(int argc, char** argv)
{
	clock_t begin, end;
	begin = clock();
	std::string out_file;
	std::ostream* outp = &std::cout;

	try
	{

		createContext();
		setProperties();

		time_t rawtime;
		struct tm * timeinfo;
		char buffer[80];
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer, 80, "%d-%m-%Y %H-%M-%S", timeinfo);

		vtkconfigure(std::string(buffer));

		const int number_position = a_range->GetNumberOfPoints();

		int optimal_transducer_num;
	
	/*	std::array<double, 1250> ARC_list;
		std::array<double, 1250> out_ARC_list;
		std::array<double, 1250> xy_angle;
		std::array<double, 1250> z_angle;
		std::array<float3, 1250> transducer_location;*/

		std::vector<double> ARC_list;
		std::vector<double> out_ARC_list;
		std::vector<double> xy_angle;
		std::vector<double> z_angle;
		std::vector<float3> transducer_location;
		ARC_list.resize(transducerNum);
		out_ARC_list.resize(transducerNum);
		xy_angle.resize(transducerNum);
		z_angle.resize(transducerNum);
		transducer_location.resize(transducerNum);

		std::tie(optimal_transducer_num, ARC_list, out_ARC_list, xy_angle, z_angle, transducer_location) = do_work(0, number_position);

		writefile(std::string(buffer), optimal_transducer_num, ARC_list, out_ARC_list, xy_angle, z_angle, transducer_location);

		end = clock();          // 시간설정
		cout << "수행시간 : " << ((end - begin) / CLOCKS_PER_SEC) << endl;
		return 0;
	}
	SUTIL_CATCH(
		context->get())
}


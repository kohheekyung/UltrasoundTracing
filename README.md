# UltrasoundTracing
Ray Casting/tracing of focused ultrasound using OptiX, VTK

Code implementation of ['A parallel computing approach for optimization of a single-element transducer 
position in transcranial focused ultrasound application'](https://github.com/kohheekyung/UltrasoundTracing/blob/main/poster.pdf)(Best poster award in KSTU)

### Notice
This project only involves codes regarding GPU based raytracing of ultrasound beams (Not full codes of navigation system).

May be able to get ideas of ..:
-  Modeling a transducer
-  Intersecting lines (ultrasound from a transducer) with surface mesh (skull)
-  Retrieving the coordinates of those intersection points

### Version
- OptiX SDK 6.0.0
- CUDA 10.1
- VTK 8.0.0
- Visual studio 2015

#!/usr/bin/env python3
import os

dimNames = ['x', 'y', 'z']
debug = False

def write_typemap_point(f, cName, swigType, nDim, isConstRef) :
	
	refStr=''
	constStr=''
	if isConstRef:
		refStr=' & '
		constStr=' const '
	f.write('//--------------------------------------------------------\n')
	f.write('//Typemap '+constStr+' '+cName+' '+refStr+'\n\n')
	f.write('//typecheck to allow overloaded functions or default values\n')
	f.write('//see http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_nn27\n')
	f.write('%typemap(typecheck,precedence=SWIG_TYPECHECK_BOOL) '+constStr+' '+cName+' '+refStr+' {\n')
	f.write('	//test if '+cName+'\n')
	f.write('	int res = SWIG_CheckState(SWIG_ConvertPtr($input, 0, '+swigType+', SWIG_POINTER_NO_NULL | 0));\n')
	f.write('	if (!res) //test if sequence\n')
	f.write('		res = (PySequence_Check($input)) && (PyObject_Length($input) == '+str(nDim)+');\n')
	f.write('	$1 = res;\n')
	f.write('}\n\n')
	f.write('//convert py seq of 2 scalars into '+cName+'\n')
	f.write('%typemap(in) '+constStr+' '+cName+' '+refStr+' ('+cName+' pt){\n')
	if debug:
		f.write('	f.writef("In typemap '+constStr+' '+cName+' '+refStr+'\\n");\n')
	f.write('	if (!PySequence_Check($input)) { //not a seq: must be a '+cName+'\n')
	f.write('	PyObject *args = $input;\n')
	f.write('	void *argp1 ;\n')
	f.write('	int res1 = 0 ;\n')
	f.write('	res1 = SWIG_ConvertPtr(args, &argp1, '+swigType+',	0	| 0);\n')
	f.write('	if (!SWIG_IsOK(res1)) {\n')
	f.write('		SWIG_exception_fail(SWIG_ArgError(res1), " impossible to convert argument into '+cName+'"); \n')
	f.write('	}\n')
	f.write('	if (!argp1) {\n')
	f.write('		SWIG_exception_fail(SWIG_ValueError, "invalid null reference type '+cName+'");\n')
	f.write('	} else {\n')
	if isConstRef:
		f.write('		$1 = reinterpret_cast< '+cName+' * >(argp1);\n')
	else:
		f.write('		Pt2d * temp = reinterpret_cast< Pt2d * >(argp1);\n')
		f.write('		pt = *temp;\n')
		f.write('		if (SWIG_IsNewObj(res1)) delete temp;\n')
		f.write('		$1 = pt;\n')
	f.write('	}\n')
	f.write('	}else{ //convert sequence into '+cName+'\n')
	f.write('	if (PyObject_Length($input) != '+str(nDim)+') {\n')
	f.write('		PyErr_SetString(PyExc_ValueError,"Expecting a sequence with '+str(nDim)+' elements");\n')
	f.write('		return NULL;\n')
	f.write('	}\n')
	f.write('	double temp['+str(nDim)+'];\n')
	f.write('	int i;\n')
	f.write('	for (i =0; i < '+str(nDim)+'; i++) {\n')
	f.write('		PyObject *o = PySequence_GetItem($input,i);\n')
	f.write('		if (PyFloat_Check(o)) {\n')
	f.write('		 temp[i] = PyFloat_AsDouble(o);\n')
	f.write('		}else if (PyLong_Check(o)) {\n')
	f.write('		temp[i] = PyLong_AsDouble(o);\n')
	f.write('		}else{\n')
	f.write('			Py_XDECREF(o);\n')
	f.write('		 PyErr_SetString(PyExc_ValueError,"Expecting a sequence of scalars");\n')
	f.write('		 return NULL;\n')
	f.write('		}\n')
	f.write('		temp[i] = PyFloat_AsDouble(o);\n')
	f.write('		Py_DECREF(o);\n')
	f.write('	}\n')
	for i in range(nDim):
		f.write('	pt.'+dimNames[i]+'() = temp['+str(i)+'];\n')
	f.write('	$1 = '+refStr+'pt;\n')
	f.write('	}\n')
	f.write('};\n\n')

dir_path = "tmp"
os.makedirs(dir_path, exist_ok="True")
f_typemaps = open(dir_path+"/typemaps.i", "w")
f_typemaps.write("// Auto-generated file\n\n")

write_typemap_point(f_typemaps, 'MMVII::cPt2di', 'SWIGTYPE_p_MMVII__cPtxdT_int_2_t', 2, False)
write_typemap_point(f_typemaps, 'MMVII::cPt2di', 'SWIGTYPE_p_MMVII__cPtxdT_int_2_t', 2, True)
write_typemap_point(f_typemaps, 'MMVII::cPt2dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_2_t', 2, False)
write_typemap_point(f_typemaps, 'MMVII::cPt2dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_2_t', 2, True)
write_typemap_point(f_typemaps, 'MMVII::cPt3di', 'SWIGTYPE_p_MMVII__cPtxdT_int_3_t', 3, False)
write_typemap_point(f_typemaps, 'MMVII::cPt3di', 'SWIGTYPE_p_MMVII__cPtxdT_int_3_t', 3, True)
write_typemap_point(f_typemaps, 'MMVII::cPt3dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_3_t', 3, False)
write_typemap_point(f_typemaps, 'MMVII::cPt3dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_3_t', 3, True)

f_typemaps.close()

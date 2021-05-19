#!/usr/bin/env python3

dimNames = ['x', 'y', 'z']
debug = False

def write_typemap_point(cName, swigType, nDim, isConstRef) :
	
	refStr=''
	constStr=''
	if isConstRef:
		refStr=' & '
		constStr=' const '
	print('//--------------------------------------------------------')
	print('//Typemap '+constStr+' '+cName+' '+refStr)
	print()
	print('//typecheck to allow overloaded functions or default values')
	print('//see http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_nn27')
	print('%typemap(typecheck,precedence=SWIG_TYPECHECK_BOOL) '+constStr+' '+cName+' '+refStr+' {')
	print('	//test if '+cName);
	print('	int res = SWIG_CheckState(SWIG_ConvertPtr($input, 0, '+swigType+', SWIG_POINTER_NO_NULL | 0));')
	print('	if (!res) //test if sequence')
	print('		res = (PySequence_Check($input)) && (PyObject_Length($input) == '+str(nDim)+');')
	print('	$1 = res;')
	print('}')
	print()
	print('//convert py seq of 2 scalars into '+cName)
	print('%typemap(in) '+constStr+' '+cName+' '+refStr+' ('+cName+' pt){')
	if debug:
		print('	printf("In typemap '+constStr+' '+cName+' '+refStr+'\\n");')
	print('	if (!PySequence_Check($input)) { //not a seq: must be a '+cName)
	print('	PyObject *args = $input;')
	print('	void *argp1 ;')
	print('	int res1 = 0 ;')
	print('	res1 = SWIG_ConvertPtr(args, &argp1, '+swigType+',	0	| 0);')
	print('	if (!SWIG_IsOK(res1)) {')
	print('		SWIG_exception_fail(SWIG_ArgError(res1), " impossible to convert argument into '+cName+'"); ')
	print('	}')
	print('	if (!argp1) {')
	print('		SWIG_exception_fail(SWIG_ValueError, "invalid null reference type '+cName+'");')
	print('	} else {')
	if isConstRef:
		print('		$1 = reinterpret_cast< '+cName+' * >(argp1);')
	else:
		print('		Pt2d * temp = reinterpret_cast< Pt2d * >(argp1);')
		print('		pt = *temp;')
		print('		if (SWIG_IsNewObj(res1)) delete temp;')
		print('		$1 = pt;')
	print('	}')
	print('	}else{ //convert sequence into '+cName+'')
	print('	if (PyObject_Length($input) != '+str(nDim)+') {')
	print('		PyErr_SetString(PyExc_ValueError,"Expecting a sequence with '+str(nDim)+' elements");')
	print('		return NULL;')
	print('	}')
	print('	double temp['+str(nDim)+'];')
	print('	int i;')
	print('	for (i =0; i < '+str(nDim)+'; i++) {')
	print('		PyObject *o = PySequence_GetItem($input,i);')
	print('		if (PyFloat_Check(o)) {')
	print('		 temp[i] = PyFloat_AsDouble(o);')
	print('		}else if (PyLong_Check(o)) {')
	print('		temp[i] = PyLong_AsDouble(o);')
	print('		}else{')
	print('			Py_XDECREF(o);')
	print('		 PyErr_SetString(PyExc_ValueError,"Expecting a sequence of scalars");')
	print('		 return NULL;')
	print('		}')
	print('		temp[i] = PyFloat_AsDouble(o);')
	print('		Py_DECREF(o);')
	print('	}')
	for i in range(nDim):
		print('	pt.'+dimNames[i]+'() = temp['+str(i)+'];')
	print('	$1 = '+refStr+'pt;')
	print('	}')
	print('};')
	print()

write_typemap_point('MMVII::cPt2di', 'SWIGTYPE_p_MMVII__cPtxdT_int_2_t', 2, False)
write_typemap_point('MMVII::cPt2di', 'SWIGTYPE_p_MMVII__cPtxdT_int_2_t', 2, True)
write_typemap_point('MMVII::cPt2dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_2_t', 2, False)
write_typemap_point('MMVII::cPt2dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_2_t', 2, True)
write_typemap_point('MMVII::cPt3di', 'SWIGTYPE_p_MMVII__cPtxdT_int_3_t', 3, False)
write_typemap_point('MMVII::cPt3di', 'SWIGTYPE_p_MMVII__cPtxdT_int_3_t', 3, True)
write_typemap_point('MMVII::cPt3dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_3_t', 3, False)
write_typemap_point('MMVII::cPt3dr', 'SWIGTYPE_p_MMVII__cPtxdT_double_3_t', 3, True)

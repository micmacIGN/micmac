#!/usr/bin/env python3

#fix http://www.swig.org/Doc4.0/Python.html#Python_memory_management_member_variables

def write_backref(back_name, backClass, function_signature) :
	print("""
	%fragment(\""""+back_name+"""_reference_init\", \"init\") {
	  // Thread-safe initialization - initialize during Python module initialization
	  """+back_name+"""_reference();
	}
	%fragment(\""""+back_name+"""_reference_function\", \"header\", fragment=\""""+back_name+"""_reference_init\") {
	  static PyObject *"""+back_name+"""_reference() {
	    static PyObject *"""+back_name+"""_reference_string = SWIG_Python_str_FromChar(\"_"""+back_name+"""_reference\");
	    return """+back_name+"""_reference_string;
	  }
	}
	%extend """+backClass+""" {
	  // A reference to the parent class is added to ensure the underlying C++
	  // object is not deleted while the item is in use
	  %typemap(ret, fragment=\""""+back_name+"""_reference_function\") """+function_signature+""" %{
	    PyObject_SetAttr($result, """+back_name+"""_reference(), $self);
	  %}
	}
	
	""")
	
write_backref("cAimePCar", "MMVII::cAimePCar", "const MMVII::cPtxd<double,2>&   Pt")



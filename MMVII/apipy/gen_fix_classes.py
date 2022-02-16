#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#pip3 install libclang
#clang++ -cc1 -ast-dump my_source.cpp
#https://libclang.readthedocs.io/en/latest/#
#https://sudonull.com/post/907-An-example-of-parsing-C-code-using-libclang-in-Python
#https://github.com/tekord/Python-Clang-RTTI-Generator-Example
#https://jonasdevlieghere.com/understanding-the-clang-ast/

import os
import clang.cindex
from clang.cindex import CursorKind as ck
from clang.cindex import TypeKind as tk

all_headers = [
  "MMVII_enums.h", "MMVII_Ptxd.h", "MMVII_Images.h",
  "MMVII_memory.h", "MMVII_nums.h", "MMVII_AimeTieP.h"
]

#all_headers = [ "MMVII_Images.h" ]

path = "../include"

cpp_str = ""
for h in all_headers:
  cpp_str += f'#include "{path:}/{h:}"\n'

print("Cpp source:")
print(cpp_str)

index = clang.cindex.Index.create()
#translation_unit
tu = index.parse('tmp.cpp', args=['-std=c++11', '-fopenmp'],  
                unsaved_files=[('tmp.cpp', cpp_str)],  options=0)

def filter(nodes, kinds, recursive = False, verbose = False):
  result = []
  for i in nodes:
    if verbose:
      print("test node ", i.spelling, i.kind, end="")
      if i.kind in kinds:
        print(" selected!")
      else:
        print()
    if i.kind in kinds:
      result.append(i)
    if recursive:
      result += filter(i.get_children(), kinds, True, verbose)
  return result

def typeFromTokens(cursor, name):
  res = ""
  tokens = [j.spelling for j  in cursor.get_tokens()]
  type_tokens = []
  for t in tokens:
    if t == name:
      break
    type_tokens.append(t)
  return " ".join(type_tokens)

class CppMethod(object):
  def __init__(self, cursor):
    self.name = cursor.spelling
    self.res_ref = cursor.result_type.kind == tk.LVALUEREFERENCE
    self.res_type = cursor.result_type.spelling #does not work for template types
    if self.name:
      self.res_type = typeFromTokens(cursor, self.name)
    self.is_const = cursor.is_const_method()
    self.is_shadowed = False
  def __str__(self):
    str_const = "const" if self.is_const else ""
    str_ref = "[ref]" if self.res_ref else ""
    return f"{self.res_type} {str_ref} {self.name}() {str_const}"

class CppAttribute(object):
  def __init__(self, cursor):
    self.name = cursor.spelling
    self.type = cursor.type.spelling #does not work for template type
    if self.name:
      self.type = typeFromTokens(cursor, self.name)
    self.type_nonamespace = self.type.split("::")[-1]
  def __str__(self):
    return f"{self.type} {self.name}"

class CppClass(object):
  def __init__(self, cursor):
    self.name = cursor.lexical_parent.spelling + "::" + cursor.spelling
    self.methods = []
    self.attributes = []
    self.empty = True
    all_methods = filter(cursor.get_children(), [ck.CXX_METHOD])
    all_attributs = filter(cursor.get_children(), [ck.FIELD_DECL])
    for j in all_methods:
      self.methods.append(CppMethod(j))
    for j in all_attributs:
      self.attributes.append(CppAttribute(j))
    self.empty = not (self.methods or self.attributes)
  def __str__(self):
    out = "Class "+self.name+"\n"
    out += "  Methods:\n"
    for m in self.methods:
      out += "   - "+str(m)+"\n"
    out += "  Attributes:\n"
    for a in self.attributes:
      out += "   - "+str(a)+"\n"
    return out


# ck.CLASS_TEMPLATE for template classes, but how to predic their used types?
all_classes_cursor = filter(tu.cursor.get_children(), [ck.CLASS_DECL, ck.STRUCT_DECL, ck.CLASS_TEMPLATE], True, False)

dir_path = "tmp"
os.makedirs(dir_path, exist_ok="True")
f_ignore = open(dir_path+"/ignore_nonconst_overloading.i", "w")
f_rename = open(dir_path+"/rename_nonref.i", "w")
f_nonref = open(dir_path+"/return_nonref.i", "w")
f_include = open(dir_path+"/h_to_include.i", "w")

f_ignore.write("// Auto-generated file\n\n")
f_rename.write("// Auto-generated file\n\n")
f_nonref.write("// Auto-generated file\n\n")
f_include.write("// Auto-generated file\n\n")

for h in all_headers:
  f_include.write(f'%include "{h}"\n')

f_include.close()

for v in all_classes_cursor:
  c = CppClass(v)
  if c.empty:
    continue
  print(c)
  
  #search for duplicate methods const/non-const
  for i in range(len(c.methods)):
    for j in range(i+1,len(c.methods)):
      if c.methods[i].name == c.methods[j].name:
        print("Duplicate: ",c.methods[i], "/", c.methods[j])
        f_ignore.write(f"%ignore {c.name}::{c.methods[j].name}();\n")
        if not c.methods[i].is_const:
          c.methods[i].is_shadowed = True
        elif not c.methods[j].is_const:
          c.methods[j].is_shadowed = True
  
  #search for getter returning ref
  for m in c.methods:
    if m.res_ref and not m.is_shadowed:
      for a in c.attributes:
        if a.name == "m"+m.name and (a.type + " &" == m.res_type or "const " + a.type + " &" == m.res_type)  :
          str_const = "const" if m.is_const else ""
          print("Getter ref:", a, "/", m)
          f_rename.write(f'%ignore {c.name}::{m.name}() {str_const};\n')
          f_rename.write(f'%rename("{m.name}") {c.name}::{m.name}_py() {str_const};\n')
          f_nonref.write(f'%extend {c.name} {{ {a.type} {m.name}_py() {str_const} {{ return $self->{m.name}(); }} }}\n\n')
  print()

f_ignore.close()
f_rename.close()
f_nonref.close()

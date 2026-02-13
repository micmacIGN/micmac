#!/bin/bash
# PATH to 'dumpbin.exe', included in Microsoft Visual Studio
DB='/d/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/dumpbin.exe'


# Extract first level DLLs used by program or dll in $1
db(){
"$DB" -DEPENDENTS $1 | awk 'BEGIN {doit=0} /^ *Image/ {doit=2} /^ *Summary/ {doit=0} { if (doit==1&& NF == 1 ) print $1; if (doit==2) doit=1 }'
}


missing=
seen=
# fill "list" with all DLLs used by $1 (recurse)
get_missing(){
  dlls=$(db $1)
  for dll in $dlls
  do
    if [[ "$seen" == *"$dll"* ]] ; then		# already processed
      continue
    fi
    seen="$seen $dll"
    if [ -f $dll ] ; then
      get_missing $dll				# if DLL file exist, recurse in it
    else
      if ! type $dll >/dev/null 2>&1 ; then
        missing="$missing $dll"    		# else, and if not in the path, add it to missing list
      fi
    fi
  done
}

get_missing $1

# print missing DLLs,
for dll in "$missing"
do
  echo "'$dll' is missing"
done

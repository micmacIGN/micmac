#!/bin/bash 

if  [ -n "$1" -a "$1" != "-f" ]
then
  echo "Usage: $(basename $0)  [-f]" 
  echo "   Print MMVII source lines using '\n' with StdOut() instead of std::endl."
  echo "   With -f, fix these lines by replacing '\n' with std::endl."
  exit 0
fi

if [ ! -d bin -o ! -d src -o ! -d include -o ! -d MMVII-RessourceDir -o ! -d MMVII-UseCaseDataSet ]
then
  echo "Please, change current directory to the main MMVII directory."
  exit 1
fi

if [ "$1" != "-f" ]
then
  grep -n -r --include "*.cpp" --include "*.h" -v std::endl | grep --color=always 'StdOut().*\\\n['"'"'"]' 
  exit 0
fi

if !  git diff-index --quiet HEAD --
then
  echo "Please, commit or revert existing modifications before fixing StdOut() << std::endl."
  exit 1
fi

cd include || exit
find . -name "*.h" -exec sed -i -e '/std::endl/!s/\(StdOut().*<< *\)['"'"'"]\\n['"'"'"]/\1std::endl/' {} \;
find . -name "*.h" -exec sed -i -e '/std::endl/!s/\(StdOut().*\)\\n"/\1" << std::endl/' {} \;

cd ../src || exit
find . -name "*.h" -exec sed -i -e '/std::endl/!s/\(StdOut().*<< *\)['"'"'"]\\n['"'"'"]/\1std::endl/' {} \;
find . -name "*.h" -exec sed -i -e '/std::endl/!s/\(StdOut().*\)\\n"/\1" << std::endl/' {} \;
find . -name "*.cpp" -exec sed -i -e '/std::endl/!s/\(StdOut().*<< *\)['"'"'"]\\n['"'"'"]/\1std::endl/' {} \;
find . -name "*.cpp" -exec sed -i -e '/std::endl/!s/\(StdOut().*\)\\n"/\1" << std::endl/' {} \;

if git diff --exit-code
then
  echo "Done. No modifications."
else
  echo "Done. Dont't forget to recompile/test before pushing the modifications ..."
fi

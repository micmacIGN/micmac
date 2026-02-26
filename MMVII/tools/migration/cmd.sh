rm -fr MMVII

case "$1" in
  g*) rm -fr micmac ;  git clone git@github.com:micmacIGN/micmac.git ;;
  h*) rm -fr micmac ;  git clone https://github.com/micmacIGN/micmac.git ;;
  *) ;; 
esac

cp -a  micmac MMVII || exit 1
cd MMVII
../git-filter-repo --replace-refs delete-no-add --subdirectory-filter MMVII

../tabs.py

git tag -l "*v1*" | xargs -r git tag -d

git branch -m master main

mkdir -p .github/workflows/
cp ../build_mmvii.yml .github/workflows/
git add .github/workflows/build_mmvii.yml
git commit -m "Build MMVII on GitHub"

patch <../CMakeLists.patch
git commit -am "CMakeLists for outside mm3d repository"

git apply ../mm3dbin.patch
git commit -am "handle mm3d binary"

git remote add origin https://github.com/meynardc/mm2.git

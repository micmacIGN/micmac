rm -fr MMVII

case "$1" in
  g*) rm -fr micmac ;  git clone git@github.com:micmacIGN/micmac.git ;;
  h*) rm -fr micmac ;  git clone https://github.com/micmacIGN/micmac.git ;;
  *) ;; 
esac

cp -a  micmac MMVII || exit 1
cd MMVII
../git-filter-repo --replace-refs delete-no-add --subdirectory-filter MMVII --prune-degenerate always
git branch -d newFeature_PIMsResolTerrain AjoutMNTInitMalt GIMMI Yilin drunk_ori dev_Tristan  MeshForMeMo  test_swig_py_file_marche_pas CroBA  Sat3D  ExportPlyGroundCrop test_swig_py_er histopipe Option_Spatial CompilCentOS  AimePy  er  RobustBlinis yann forSAT4GEO_Testing forSAT4GEO  apib11-jm  apib11 test_swig_py  ch_hiatus er-apyb11 BlocRigid am-test am-next jm-topo cm-next am_fdsc  test_conditioning 
git branch -df cm-apyb11 images_gdal cm_polyser clino IncludeALGLIB
git branch --list 'dependabot/*' | xargs -r git branch -df

git-filter-repo  --replace-refs delete-no-add  --path DocInternet/ --path src/aux.cpp  --path src/CodedTarget/CodedTarget --path src/CodedTarget/Exemples --invert-paths

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

echo "delete depandabot branch"
echo "push -f --all"

git remote add origin https://github.com/micmacv2/MMVII.git

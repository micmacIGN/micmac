rm -fr MMVII MMVII-UseCaseDataSet

case "$1" in
  g*) rm -fr micmac ;  git clone git@github.com:micmacIGN/micmac.git ;;
  h*) rm -fr micmac ;  git clone https://github.com/micmacIGN/micmac.git ;;
  *) ;; 
esac

cp -a  micmac MMVII || exit 1
cp -a  MMVII/MMVII/MMVII-UseCaseDataSet .
cd MMVII-UseCaseDataSet
git init
git add *
git commit -a -m "Initial commit"
git remote add origin https://github.com/micmac-V2/MMVII-UseCaseDataSet.git

cd ..
cd MMVII
../git-filter-repo --replace-refs delete-no-add --subdirectory-filter MMVII --prune-degenerate always
git branch -d newFeature_PIMsResolTerrain AjoutMNTInitMalt GIMMI Yilin drunk_ori dev_Tristan  MeshForMeMo  test_swig_py_file_marche_pas CroBA  Sat3D  ExportPlyGroundCrop test_swig_py_er histopipe Option_Spatial CompilCentOS  AimePy  er  RobustBlinis yann forSAT4GEO_Testing forSAT4GEO  apib11-jm  apib11 test_swig_py  ch_hiatus er-apyb11 BlocRigid am-test am-next jm-topo cm-next am_fdsc  test_conditioning 
git branch -df cm-apyb11 images_gdal cm_polyser clino IncludeALGLIB
git branch --list 'dependabot/*' | xargs -r git branch -df

git-filter-repo  --replace-refs delete-no-add  --path DocInternet/ --path src/aux.cpp  --path src/CodedTarget/CodedTarget --path src/CodedTarget/Exemples --path MMVII-UseCaseDataSet --invert-paths
git-filter-repo  --replace-refs delete-no-add  --path MMVII-TestDir/Input/EPIP/Tiny/Px_ImL.tif --path MMVII-TestDir/Input/EPIP/Tiny/RefPx.tif --path Doc/Paper/Epipolar_ipol/Epipolar_ipol.pdf  --invert-paths
git-filter-repo  --replace-refs delete-no-add  --path src/LearningMatching/trained_model_assets/UNET32/OCC_AWARE_PANACHE_NORMED_UNET_FEATURES_AERIAL.pt --path src/LearningMatching/trained_model_assets/UNET_ATTENTION/OCC_AWARE_PANACHE_NORMED_UNET_ATTENTION_FEATURES_AERIAL.pt --path src/DenseMatch/RAFT-Stereo/models/raftstereo-realtime.pth --path src/DenseMatch/RAFT-Stereo/models/iraftstereo_rvc.pth --path  src/DenseMatch/RAFT-Stereo/models/raftstereo-eth3d.pth --path src/DenseMatch/RAFT-Stereo/models/raftstereo-middlebury.pth --path src/DenseMatch/RAFT-Stereo/models/raftstereo-sceneflow.pth --path src/DenseMatch/PSMNet/models/finetune_PSMnet.tar --path src/DenseMatch/RAFT-Stereo/models/270000_raftstereo_experiment-from-previous-train-150K_same_steplr30K.pth --path src/DenseMatch/RAFT-Stereo/models/1000002_epoch_raftstereo_experiment-PATCH-640.pth.gz --path src/DenseMatch/RAFT-Stereo/models/375002_epoch_raftstereo_experiment.pth.gz --path src/LearningMatching/trained_model_assets/MSAFF/OCC_AWARE_PANACHE_NORMED_MSAFF_FEATURES_AERIAL.pt  --invert-paths

git filter-repo --replace-refs delete-no-add  --path-glob '*.o' --path-glob '*.d' --invert-paths

../tabs.py

git tag -l "*v1*" | xargs -r git tag -d

git branch -m master main
git reflog expire --expire=now --all
git gc --prune=now --aggressive

mkdir -p .github/workflows/
cp ../build_mmvii.yml .github/workflows/
cp ../README.md .
cp ../vMMVII_CMakeLists.txt vMMVII/CMakeLists.txt
git add .github/workflows/build_mmvii.yml
patch <../CMakeLists.patch
git apply ../mm3dbin.patch
git commit -am "Patches for MMVII new repository"
git remote add origin https://github.com/micmac-v2/MMVII.git

git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '/^blob/ { printf "%10d KB\t%s\n", $3/1024, $4 }' \
  | sort -rn \
  | head -n 20

echo 
echo "Remaining task: push -f --all"
echo "  dans MMVII et MMVII-UseCaseDataSet"

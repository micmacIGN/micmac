#ifndef _MESH_DEV_H_
#define _MESH_DEV_H_

namespace MMVII
{



class cMeshDev_BestIm
{
    public :
       std::vector<std::string> mNames;
       std::vector<int>         mNumBestIm;
       std::vector<double>      mBestResol;

};

//void AddData(cAuxAr2007  anAux,cMeshDev_BestIm& aRMS);

extern const std::string  MeshDev_NameTriResol;

}

#endif  //  _MESH_DEV_H_

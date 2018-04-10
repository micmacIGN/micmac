#ifndef _ELISE_TAPAS_CAMPARI_H_
#define _ELISE_TAPAS_CAMPARI_H_

/*
   This file, who should have exist far before, contains some devlopment common to Tapas & Campari
*/


class cAppli_Tapas_Campari
{
    public :
       cAppli_Tapas_Campari();
       void AddParamBloc(std::string & mCom);
       LArgMain &     ArgATP();

    protected :
    private :
       bool                      mWithBlock;
       std::string               mNameInputBloc;
       std::string               mNameOutputBloc;
       std::vector<std::string>  mVBlockRel;
       std::vector<std::string>  mVBlockGlob;
       std::vector<std::string>  mVBlockDistGlob;
       std::vector<std::string>  mVOptGlob;
    private :
       void AddParamBloc(std::string & mCom,std::vector<std::string> & aVBL,const std::string & aPref,bool ModeRot);
       LArgMain                  *mArg;
};


#endif //  _ELISE_TAPAS_CAMPARI_H_

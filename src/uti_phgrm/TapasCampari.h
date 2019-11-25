#ifndef _ELISE_TAPAS_CAMPARI_H_
#define _ELISE_TAPAS_CAMPARI_H_

/*
   This file, who should have exist far before, contains some devlopment common to Tapas & Campari
*/

std::string BlQUOTE (const std::string &) ;


class cAppli_Tapas_Campari
{
    public :
       cAppli_Tapas_Campari();
       void AddParamBloc(std::string & mCom);
       LArgMain &     ArgATP();

       bool                        mWithBlock;
       std::string ExtendPattern(const std::string & aPatGlob,const std::string & aPatCenter,cInterfChantierNameManipulateur *);
       std::string TimeStamp(const std::string & aName,cInterfChantierNameManipulateur * anICNM);
       const cStructBlockCam &  SBC() const;

       void InitAllImages(const std::string & aPat,cInterfChantierNameManipulateur * anICNM);

       void InitVerifModele(const std::string & aMod,cInterfChantierNameManipulateur *);
       std::list<std::string> GetAuthorizedModel();
       void SyncLocAndGlobVar();

       const std::vector<std::string> & BlocImagesByTime() const;
       const std::vector<std::string> & BlocTimeStamps() const;
       std::map<std::string,int> & BlocCptTime() ;
       const std::string & StrParamBloc() const;
       int   NbInBloc() const;
       int  LongestBloc(int aK0,int aK1);  // Plus grand sequence a time stamp = dans [K0,K1[
       std::string eModAutom;

       int  LocDegGen;
       bool LocLibDec;
       bool LocLibCD;
       int  LocDRadMaxUSer;
       bool LocLibPP ;
       bool LocLibFoc;
       int LocDegAdd;
       bool IsAutoCal;
       bool IsFigee;
       double PropDiag;
       bool GlobLibAff;
       bool GlobLibDec;
       bool GlobLibPP;
       bool GlobLibCD;
       bool GlobLibFoc;
       int  GlobDRadMaxUSer;
       int  GlobDegGen;
       int GlobDegAdd;

       bool ModeleAdditional;
       bool ModeleAddFour;
       bool ModeleAddPoly;
       std::string TheModelAdd;
       std::string mSauvAutom;
       double      mRatioMaxDistCS;

       bool UseRappOnZ();


       const int  & DSElimB () const {return  mDSElimB;}
       bool ExportMatrixMarket()  const {return mExportMatrixMarket;}
    private :
      
       std::string               mStrParamBloc;
       std::string               mNameInputBloc;
       std::string               mNameOutputBloc;
       std::vector<std::string>  mVBlockRel;
       std::vector<std::string>  mVBlockGlob;
       std::vector<std::string>  mVBlockDistGlob;
       std::vector<std::string>  mVOptGlob;
       cStructBlockCam           mSBC;
       bool                      mNamesBlockInit;
       std::vector<std::string>  mBlocImagesByTime;
       std::vector<std::string>  mBlocTimeStamps;
       std::map<std::string,int> mBlocCptTime;
       std::vector<std::string>  mRapOnZ;



    private :
      // ModePose = ! ModeDist
       void AddParamBloc(std::string & mCom,std::vector<std::string> & aVBL,const std::string & aPref,bool ModePose);
       int      mDSElimB;
       bool     mExportMatrixMarket;
       LArgMain                  *mArg;
};


#endif //  _ELISE_TAPAS_CAMPARI_H_

#ifndef CERO_APPLI_H
#define CERO_APPLI_H
#include "cero_modelonepaire.h"

class cERO_Appli
{
public:
     cERO_Appli(int argc, char** argv);

    void computeImCplOverlap();
    void computeModel4EveryPairs();
    // charge la liste de paire renseignée par l'utilisateur
    void loadImPair();
    void applyRE();

   private:
     cInterfChantierNameManipulateur * mICNM;
     // liste de couple d'images
     cSauvegardeNamedRel mSNR ;
     std::string mFileClpIm, mDirOut,mDir,mFullName,mFileOutModels;
     std::list<std::string> mLFile;
     bool mDebug,mSaveSingleOrtho;
     std::string mPatOrt;
     std::string mPatPrio; // les images qu'on ne souhaite pas égaliser.
     std::vector<cImGeo> mLIm;
     std::vector<c2DLineModel> mL2Dmod;// liste des modèles linéaires, un pour chaque images chargées.

     int mMinOverX_Y,mMinOverX_Y_fichierCouple,mPropPixRec;
     void moyenneModelPerOrt();
     void loadEROSmodel4OneIm(string aNameOrt);
     void saveModelsGlob();// one model per im
};



#endif // CERO_APPLI_H

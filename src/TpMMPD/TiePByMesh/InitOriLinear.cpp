#include <stdio.h>
#include "StdAfx.h"
#include "InitOriLinear.h"

SerieCamLinear::SerieCamLinear(string aPatImgREF, string aPatImgNEW, string aOri, string aOriOut, int index)
{
    this->mOriOut = aOriOut;
    this->mIndexCam = index;
    this->mPatImgNEW = aPatImgNEW;
    this->mPatImgREF = aPatImgREF;
    this->mOri = aOri;
    string aDirNEW,aDirREF, aPatNEW,aPatREF;
    SplitDirAndFile(aDirNEW, aPatNEW, mPatImgNEW);
    SplitDirAndFile(aDirREF, aPatREF, mPatImgREF);
    this->mICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirNEW);
    this->mSetImgNEW = *(mICNM->Get(mPatImgNEW));
    this->mSetImgREF = *(mICNM->Get(mPatImgREF));
    string aDirOri;
    for (uint i=0; i<mSetImgREF.size(); i++)
    {
        aDirOri= mOri + "/Orientation-"+mSetImgREF[i]+".xml";
        bool Exist = ELISE_fp::exist_file(aDirOri);
        if (!Exist)
            StdCorrecNameOrient(mOri,aDirREF);
        aDirOri= mOri + "/Orientation-"+mSetImgREF[i]+".xml";
        cOrientationConique aOriFile=StdGetFromPCP(aDirOri,OrientationConique);
        this->mSetOriREF.push_back(aOriFile);
    }
    cout<<"Cam has: "<<endl;
    cout<<" ++ "<<mSetImgREF.size()<<" imgs REF"<<endl;
    for (uint i=0; i<mSetImgREF.size(); i++)
        cout<<"     ++"<<mSetImgREF[i]<<endl;
    cout<<" ++ "<<mSetImgNEW.size()<<" imgs NEW"<<endl;
    for (uint i=0; i<mSetImgNEW.size(); i++)
        cout<<"     ++"<<mSetImgNEW[i]<<endl;
}


void SerieCamLinear::calPosRlt()
{
    cout<<"Cal Position Relative of Cam "<<this->mIndexCam<<endl;
    if (mSystem.size() == 0)
        cout<<"ERROR : mSystem havent' had any cam !"<<endl;
    else
    {
        cOrientationConique aOriREFImg0ThisCam = this->mSetOriREF[0];
        for(uint i=0; i<mSystem.size(); i++)
        {
          SerieCamLinear * cam = mSystem[i];
          cOrientationConique aOriREFImg0OtherCam = cam->mSetOriREF[0];
          Pt3dr posRlt = aOriREFImg0OtherCam.Externe().Centre() - aOriREFImg0ThisCam.Externe().Centre();
          this->posRltWithOtherCam.push_back(posRlt);
          cout<<" ++Cam "<<cam->mIndexCam<<" : "<<posRlt<<endl;
        }
    }
}

Pt3dr SerieCamLinear::calVecMouvement()
{
    cout<<"Cal Vector Deplacement Cam "<<this->mIndexCam;
    Pt3dr result;
    cOrientationConique aOriImg0 = mSetOriREF[0];
    Pt3dr CentreImg0 = aOriImg0.Externe().Centre();
    Pt3dr acc(0,0,0);
    for (uint i=1; i<this->mSetOriREF.size(); i++)
    {
        cOrientationConique aOriImg = mSetOriREF[i];
        Pt3dr CentreImg = aOriImg.Externe().Centre();
        acc = acc + CentreImg - CentreImg0;
        CentreImg0 = CentreImg;
    }
    result = acc/(this->mSetOriREF.size()-1);
    this->mVecMouvement = result;
    cout<<" "<<mVecMouvement<<endl;
    return result;
}

void SerieCamLinear::initSerie(Pt3dr vecMouvCam0 , vector<string> aVecPoseTurn, vector<double> aVecAngleTurn)
{
    cout<<"Init serie Cam : "<<this->mIndexCam<<endl;
    cOrientationConique aOriLastImg = this->mSetOriREF.back();
    for (uint i=0; i<this->mSetImgNEW.size(); i++)
    {
        cOrientationConique aOriInitImg = aOriLastImg;
        aOriInitImg.Externe().Centre() = aOriLastImg.Externe().Centre() + this->mVecMouvement;
        string aOriInitImgXML = this->mOriOut + "/Orientation-"+this->mSetImgNEW[i]+".xml";
        MakeFileXML(aOriInitImg, aOriInitImgXML);
        aOriLastImg = aOriInitImg;
        cout<<" ++ "<<aOriInitImg.Externe().Centre()<<endl;
        cout<<" ++ Write: "<<aOriInitImgXML<<endl;
        this->mSetOriNEW.push_back(aOriInitImg);
    }
}

void SerieCamLinear::initSerieByRefSerie(SerieCamLinear* REFSerie)
{
    cout<<"Init serie Cam : "<<this->mIndexCam<<endl;
    Pt3dr vecPosRlt = REFSerie->posRltWithOtherCam[this->mIndexCam];
    vector<cOrientationConique> aSetOriNEWSerieRef = REFSerie->mSetOriNEW;
    for (uint i=0; i<this->mSetImgNEW.size(); i++)
    {
        cOrientationConique oriRef = aSetOriNEWSerieRef[i];
        cOrientationConique oriNew = oriRef;
        oriNew.Externe().Centre() = oriRef.Externe().Centre() + vecPosRlt;
        string aOriInitImgXML = this->mOriOut + "/Orientation-"+this->mSetImgNEW[i]+".xml";
        MakeFileXML(oriNew, aOriInitImgXML);
        cout<<" ++ "<<oriNew.Externe().Centre()<<endl;
        cout<<" ++ Write: "<<aOriInitImgXML<<endl;
        this->mSetOriNEW.push_back(oriNew);
    }
}




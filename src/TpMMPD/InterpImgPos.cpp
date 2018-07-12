/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


#include "StdAfx.h"
#include "spline.h"

class cIIP_Appli
{
public :
    cIIP_Appli(int argc,char ** argv);
    void WriteImTmInFile(std::string aOutputNameFile,const std::vector<Pt3dr> & aPos,const std::vector<Pt3dr> & aVInc,const std::vector<double> & aVEcart,bool aFormat);
private :

    double ConvertLocTime(const double aT) const
    {
        return (aT-mT0Gps) * mTimeUnit;
    }
    Pt3dr   TrajetGps(int aK0,int aK1);
    Pt3dr   VitesseGps(int aK0,int aK1);
    double  TempsEcoule(int aK0,int aK1);

    Pt3dr  ToProj(const Pt3dr & aP) const;
    Pt3dr GpsInc(int aK) const;

    // Remet l'indice dans le bon intervalle
    const cOneGpsDGF & GpsSafe(int aK) const
    {
        return mDicoGps.OneGpsDGF().at(ElMax(0,ElMin(aK,int(mDicoGps.OneGpsDGF().size()-1))));
    }


    int GetIndGpsBefore(double aTime);
    std::string mDir;
    std::string mGpsFile;
    std::string mTMFile;


    // Donne reformatees pour spline
    std::vector<double>        mVX;
    std::vector<double>        mVY;
    std::vector<double>        mVZ;
    std::vector<double>        mVT;

    // Donnees d'entree
    cDicoGpsFlottant             mDicoGps;
    cDicoImgsTime                mDicoIm;

    double                       mT0Gps;
    double                       mTimeUnit;
    bool                         mModeSpline;
    int                          mNbGps;
    // cSysCoord  *                 mSysProj;
    cChSysCo   *                 mChgSys;
    bool                         mWithWPK;
    bool                         mWithInc;
    bool                         mWithIncVitesse;
    bool                         mEcart;
    bool                         mSysGeoC2Rtl;
    std::string                  mNameChSys;
    bool                         mDiscardImTM;
};


Pt3dr  cIIP_Appli::GpsInc(int aK) const
{
    return  Sup(GpsSafe(aK).Incertitude(),GpsSafe(aK+1).Incertitude());
}

Pt3dr   cIIP_Appli::ToProj(const Pt3dr & aP) const
{
    if (! mChgSys) return aP;
    return mChgSys->Src2Cibl(aP);
}

bool CmpGpsOnTime(const cOneGpsDGF & aGps1,const cOneGpsDGF & aGps2) {return aGps1.TimePt() < aGps2.TimePt() ;}
bool CmpImOnTime(const cCpleImgTime & aI1,const cCpleImgTime & aI2) {return aI1.TimeIm() < aI2.TimeIm() ;}


Pt3dr    cIIP_Appli::TrajetGps(int aK0,int aK1)
{
    return ToProj(mDicoGps.OneGpsDGF()[aK1].Pt()) - ToProj(mDicoGps.OneGpsDGF()[aK0].Pt());
}
double  cIIP_Appli::TempsEcoule(int aK0,int aK1)
{
    return (mDicoGps.OneGpsDGF()[aK1].TimePt() - mDicoGps.OneGpsDGF()[aK0].TimePt()) * mTimeUnit;
}

Pt3dr    cIIP_Appli::VitesseGps(int aK0,int aK1)
{
    return TrajetGps(aK0,aK1) / TempsEcoule(aK0,aK1);
}
void cIIP_Appli::WriteImTmInFile
(
        std::string aOutputNameFile,
        const std::vector<Pt3dr> & aVPos,
        const std::vector<Pt3dr> & aVInc,
        const std::vector<double> & aVEcart,
        bool aFormat
        )
{
    FILE* aCible = NULL;
    aCible=fopen(aOutputNameFile.c_str(),"w");

    if(aFormat)
    {
        std::string aFormat = "#F=N_X_Y_Z";
        if (mWithWPK) aFormat +=  "_W_P_K";
        if (mWithInc) aFormat +=  "_Ix_Iy_Iz";
        if (mEcart) aFormat +=  "_Sigma";
        fprintf(aCible,"%s \n",aFormat.c_str());
    }

    int aK=0;
    for (auto itI=mDicoIm.CpleImgTime().begin() ; itI!=mDicoIm.CpleImgTime().end() ;itI++,aK++)
    {
        fprintf(aCible,"%s %.6f %.6f %.6f",itI->NameIm().c_str(), aVPos.at(aK).x, aVPos.at(aK).y, aVPos.at(aK).z);
        if (mWithWPK)
            fprintf(aCible," %.6f %.6f %.6f", 0.0, 0.0, 0.0);
        if (mWithInc)
            fprintf(aCible," %.6f %.6f %.6f", aVInc.at(aK).x,aVInc.at(aK).y,aVInc.at(aK).z);
        if (mEcart)
            fprintf(aCible," %.6f",aVEcart.at(aK));
        fprintf(aCible,"\n");



    }

    fclose(aCible);
}


int cIIP_Appli::GetIndGpsBefore(double aTime)
{
    for (int aK=0 ; aK<int(mDicoGps.OneGpsDGF().size()-1) ; aK++)
    {
        if (     (mDicoGps.OneGpsDGF()[aK].TimePt()<=aTime)
                 &&  (mDicoGps.OneGpsDGF()[aK+1].TimePt()>aTime)
                 )
            return aK;
    }
    ELISE_ASSERT(false,"cIIP_Appli::GetIndGpsBefore");
    return -1;
}


cIIP_Appli::cIIP_Appli(int argc,char ** argv) :
    mTimeUnit (24  * 3600),
    mModeSpline (true),
    mChgSys     (0),
    mWithWPK    (true),
    mWithInc    (true),
    mWithIncVitesse    (true),
    mEcart        (false),
    mSysGeoC2Rtl  (false),
    mDiscardImTM(false)
{
    std::cout.precision(15) ;
    std::string aOut;
    bool aAddFormat = true;

    bool mAcceptExtrapol = false;
    std::string mPatNamePly;


    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(mDir,"Directory")
                << EAMC(mGpsFile, "GPS .xml file trajectory",  eSAM_IsExistFile)
                << EAMC(mTMFile, "Image TimeMark .xml file",  eSAM_IsExistFile),
                LArgMain() << EAM(aOut,"Out",false,"Name Output File ; Def = GPSFileName-TMFileName.txt")
                << EAM(aAddFormat,"Header",false,"Add File Format at the begining fo the File ; Def=true",eSAM_IsBool)
                << EAM(mTimeUnit,"TimeU",false,"Unity for input time, def = 1 Day ")
                << EAM(mModeSpline,"ModeSpline",false,"Interpolation spline, def=true ")
                << EAM(mPatNamePly,"PatNamePly",false,"Pattern name for Ply")
                << EAM(mWithWPK,"WithAngle",false,"Generate fake angle, def=true ")
                << EAM(mWithInc,"Inc",false,"Export uncertainty, def=true")
                << EAM(mWithIncVitesse,"SpeedInc",false,"Use speed variation in uncertainty estimation,def=true ")
                << EAM(mEcart,"Ecart",false,"Generate difference between the interpolated position and the nearest GPS position, def=false")
                << EAM(mSysGeoC2Rtl,"SysGeoC2RTL",false,"Make chgs sys from geoc to RTL of first point")
                << EAM(mNameChSys,"ChSys",false,"To chang coorrdinate system")
                << EAM(mDiscardImTM,"Discard",false,"Discard images that are taken ouside time range of GPS observation, def false.\n")

                );


    if (! EAMIsInit(&mWithWPK))
        mWithWPK = mModeSpline;



    if (! EAMIsInit(&aAddFormat) & EAMIsInit(&mModeSpline))
        aAddFormat = ! mModeSpline;

    bool mExportPly=EAMIsInit(&mPatNamePly);

    if( ! EAMIsInit(&aOut))
    {
        aOut=StdPrefixGen(mGpsFile) + "-" + StdPrefixGen(mTMFile) + ".txt";
    }

    //    static cSysCoord * RTL(const Pt3dr & Ori);

    // Lecture Xml entree


    mDicoGps = StdGetFromPCP(mGpsFile,DicoGpsFlottant);
    mDicoIm = StdGetFromPCP(mTMFile,DicoImgsTime);

    mNbGps = mDicoGps.OneGpsDGF().size();

    // Etre sur que les dates sont croissante
    std::sort(mDicoGps.OneGpsDGF().begin(),mDicoGps.OneGpsDGF().end(),CmpGpsOnTime);
    std::sort(mDicoIm.CpleImgTime().begin(),mDicoIm.CpleImgTime().end(),CmpImOnTime);

    ELISE_ASSERT(mDicoGps.OneGpsDGF().size() !=0,"Empty size");

    mT0Gps = floor(mDicoGps.OneGpsDGF().front().TimePt());

    if (EAMIsInit(&mNameChSys))
    {
        mChgSys = cChSysCo::Alloc(mNameChSys,mDir);
    }
    else if (mSysGeoC2Rtl)
    {
        mChgSys = new cChSysCo(cSysCoord::GeoC(),cSysCoord::RTL(mDicoGps.OneGpsDGF().front().Pt()));
    }

    // Formatage pour la Bb spline
    for (auto itG=mDicoGps.OneGpsDGF().begin() ; itG!=mDicoGps.OneGpsDGF().end() ;itG++)
    {
        // itG->TimePt()  = ConvertLocTime(itG->TimePt());
        mVX.push_back(itG->Pt().x);
        mVY.push_back(itG->Pt().y);
        mVZ.push_back(itG->Pt().z);
        mVT.push_back(itG->TimePt());
    }

    // Affichage interval + verif Img inclus ds Gps si pas d'extrapolation
    {
        double aT0Gps = mDicoGps.OneGpsDGF().front().TimePt();
        double aTNGps = mDicoGps.OneGpsDGF().back().TimePt();
        double aT0Im  = mDicoIm.CpleImgTime().front().TimeIm();
        double aTNIm  = mDicoIm.CpleImgTime().back().TimeIm();
        printf("****************************************************************\n");
        printf("Gps_MJD[0] = %lf LocSec=%lf \n",  aT0Gps,ConvertLocTime(aT0Gps));
        printf("Gps_MJD[end] = %lf LocSec=%lf \n",aTNGps,ConvertLocTime(aTNGps) );
        printf("****************************************************************\n");
        printf("****************************************************************\n");
        printf("Img_MJD[0] = %lf LocSec=%lf \n",aT0Im,ConvertLocTime(aT0Im));
        printf("Img_MJD[end] = %lf LocSec=%lf \n",aTNIm,ConvertLocTime(aTNIm));
        printf("****************************************************************\n");

        int ct(0);
        if (mDiscardImTM){


            for( std::vector< cCpleImgTime >::iterator iT=mDicoIm.CpleImgTime().begin();
                 iT!=mDicoIm.CpleImgTime().end();)
            {
                if (!(iT->TimeIm()>aT0Gps && iT->TimeIm()<aTNGps)){
                    std::cout << "Discard image " << iT->NameIm() << " which was shooted outside GPS recording \n";
                    iT = mDicoIm.CpleImgTime().erase(iT);
                    ct++;
                } else {
                    iT++;}
            }

        }
            if (ct>0){
            aT0Im  = mDicoIm.CpleImgTime().front().TimeIm();
            aTNIm  = mDicoIm.CpleImgTime().back().TimeIm();
            std::cout << ct << " image were discarded.\n";
            printf("****************************************************************\n");
            printf("Gps_MJD[0] = %lf LocSec=%lf \n",  aT0Gps,ConvertLocTime(aT0Gps));
            printf("Gps_MJD[end] = %lf LocSec=%lf \n",aTNGps,ConvertLocTime(aTNGps) );
            printf("****************************************************************\n");
            printf("****************************************************************\n");
            printf("Img_MJD[0] = %lf LocSec=%lf \n",aT0Im,ConvertLocTime(aT0Im));
            printf("Img_MJD[end] = %lf LocSec=%lf \n",aTNIm,ConvertLocTime(aTNIm));
            printf("****************************************************************\n");
        }


        if (! mAcceptExtrapol)
        {
            ELISE_ASSERT(aT0Im>=aT0Gps,"First image TM starts before GPS Traj !");
            ELISE_ASSERT(aTNIm<=aTNGps,"Last image TM ends after GPS Traj !");
        }

        // Test epsilon machine sur les grandeur utilisees
        if (0)
        {
            double aEps = 1.0;
            while (aT0Gps != (aT0Gps+aEps))
                aEps *= 0.9;

            std::cout << "Epsilon time = " << aEps << "\n";
        }

    }

    if (mModeSpline)
    {
/*

           for (auto itI=mDicoIm.CpleImgTime().begin() ; itI!=mDicoIm.CpleImgTime().end() ;itI++)
           {
                double aTimeI = itI->TimeIm();
                Pt3dr  aPtIm (aS_x(aTimeI),aS_y(aTimeI),aS_z(aTimeI));
                aVPtIm.push_back(aPtIm);

                // int aKGps = GetIndGpsBefore(aTimeI);
                aVPtInc.push_back(GpsInc(GetIndGpsBefore(aTimeI)));
           }
       WriteImTmInFile(aOut,aVPtIm,aVPtInc,aAddFormat);
*/
    }
    {

        //make interpolation
        tk::spline aS_x;
        tk::spline aS_y;
        tk::spline aS_z;

        aS_x.set_points(mVT,mVX);
        aS_y.set_points(mVT,mVY);
        aS_z.set_points(mVT,mVZ);


        cPlyCloud aPC;
        cElRegex * aRegName = 0;
        if (EAMIsInit(&mPatNamePly))
            aRegName = new cElRegex(mPatNamePly,10);

        std::vector<Pt3dr> aVPos;
        std::vector<Pt3dr> aVInc;
        std::vector<double> aVEcart;


        for (auto itI=mDicoIm.CpleImgTime().begin() ; itI!=mDicoIm.CpleImgTime().end() ;itI++)
        {
            Pt3dr anInc(-1,-1,-1);
            Pt3dr aPosLin (0,0,0);
            Pt3dr aPos (0,0,0);
            double aEcart = 0.0;

            double aTime = itI->TimeIm();
            Pt3dr  aPtSpline (aS_x(aTime),aS_y(aTime),aS_z(aTime));

            int aK = GetIndGpsBefore(aTime);
            const cOneGpsDGF & aGpsAv = mDicoGps.OneGpsDGF()[aK];
            const cOneGpsDGF & aGpsAp = mDicoGps.OneGpsDGF()[aK+1];
            double aTGpsAv = aGpsAv.TimePt();
            double aTGpsAp = aGpsAp.TimePt();

            // Position par interpol lineaire
            double aPdsAv = (aTGpsAp-aTime) / (aTGpsAp-aTGpsAv);
            aPosLin = aGpsAv.Pt() *aPdsAv + aGpsAp.Pt() * (1-aPdsAv);

            if ( (aK<=2) || (aK>= (mNbGps-4)))
            {
            }
            else
            {
                bool Nav =  (ElAbs(aTime-aTGpsAv) < ElAbs(aTime-aTGpsAp)) ;
                int aNK1 =  Nav  ? aK : (aK+1) ; // Nearest K
                int aNK0 = aNK1-1;
                int aNK2 = aNK1+1;

                Pt3dr aNP = mDicoGps.OneGpsDGF()[aNK1].Pt();
                Pt3dr aPosParab(0,0,0);
                // Si on estime la position par trois points par interopation sur parabole
                if (1)
                {
                    std::vector<Pt2dr> aVIntX;
                    std::vector<Pt2dr> aVIntY;
                    std::vector<Pt2dr> aVIntZ;
                    for (int aDK=-1 ; aDK<=1 ; aDK++)
                    {
                        Pt3dr aP = mDicoGps.OneGpsDGF()[aNK1+aDK].Pt();
                        // Pt3dr aP = ToProj(mDicoGps.OneGpsDGF()[aNK1+aDK].Pt());
                        // double aT = mDicoGps.OneGpsDGF()[aNK1+aDK].TimePt();
                        double aT = mDicoGps.OneGpsDGF()[aNK1+aDK].TimePt() - aTime;
                        aVIntX.push_back(Pt2dr(aT,aP.x));
                        aVIntY.push_back(Pt2dr(aT,aP.y));
                        aVIntZ.push_back(Pt2dr(aT,aP.z));

                    }
                    ElPolynome<double> aXParab = LeasSqFit(aVIntX);
                    ElPolynome<double> aYParab = LeasSqFit(aVIntY);
                    ElPolynome<double> aZParab = LeasSqFit(aVIntZ);

                    aPosParab = Pt3dr(aXParab(0),aYParab(0),aZParab(0));
                    // std::cout << "CHEK INTERP " << aPosParab -  aPosLin  << "\n";
                }


                Pt3dr aV01 = VitesseGps(aNK0,aNK1);
                Pt3dr aV12 = VitesseGps(aNK1,aNK2);
                // double aT02 = TempsEcoule(aNK0,aNK2);
                // const cOneGpsDGF & aGps0  = mDicoGps.OneGpsDGF()[aNK0];
                // const cOneGpsDGF & aGps1  = mDicoGps.OneGpsDGF()[aNK1];

                Pt3dr  aDV = (aV12-aV01);
                double aDT = ElAbs(aTime - mDicoGps.OneGpsDGF()[aNK1].TimePt()) * mTimeUnit;
                Pt3dr aIncV =  Pt3dr(ElAbs(aDV.x),ElAbs(aDV.y),ElAbs(aDV.z)) * aDT;


                Pt3dr aGpsI = GpsInc(aK);
                if (mWithIncVitesse)
                    anInc = Pt3dr
                            (
                                sqrt(ElSquare(aIncV.x)+ElSquare(aGpsI.x)),
                                sqrt(ElSquare(aIncV.y)+ElSquare(aGpsI.y)),
                                sqrt(ElSquare(aIncV.z)+ElSquare(aGpsI.z))
                                );
                else
                    anInc = aGpsI;

                if (0)
                {
                    double anIncByInterp = euclid(aPosParab-aPosLin);
                    std::cout.precision(8);
                    std::cout <<  itI->NameIm() << " "
                               <<  ConvertLocTime(aTime) << " => " << aK
                                << " Inc "  << euclid(anInc)
                                << " IncV " << euclid(aIncV)
                                << " Ratio/IncV " << euclid(aIncV) / anIncByInterp
                                << " Dt=" << aDT << "\n";
                }
                if (0)
                {
                    std::cout.precision(8);
                    std::cout <<  itI->NameIm() << " "
                               << " Dt=" << aDT << " "
                               << " Splin/Parab " << euclid(aPtSpline-aPosParab)
                               << "\n";
                }

                aPos = mModeSpline ? aPtSpline : aPosParab;
                if (mChgSys)
                {
                    aPos = ToProj(aPos);
                }
                aEcart = euclid(aPos-aNP);



                if (mExportPly)
                {
                    Pt3di aColSom(0,255,0);
                    Pt3di aColInc(255,0,0);
                    Pt3di aColName(255,255,255);

                    Pt3dr aPLoc = ToProj(aPos);
                    // std::cout << "PLOC " << aPLoc << "\n";
                    double aRay=0.05;
                    aPC.AddSphere(aColSom,aPLoc,0.05,5);
                    double aL = euclid(anInc);

                    double aStep = 0.005;
                    double Exag = 10;
                    Pt3dr aDirI(0,0,1);

                    Pt3dr aP0 = aPLoc + aDirI*aRay;
                    aPC.AddSeg(aColInc,aP0,aP0+aDirI*aL*Exag,round_up(aL/aStep));

                    if (aRegName)
                    {
                        std::string aName = MatchAndReplace(*aRegName,itI->NameIm(),"$1");
                        aPC.PutString(aName,aPLoc-aDirI*aRay*2,-aDirI,Pt3dr(1,0,0),aColName,0.03,0.01,3);
                    }
                }
            }
            // std::cout << setprecision(15) << ConvertLocTime(aTime) << " => " << aK << "\n";
            // std::cout << "\n";

            aVPos.push_back(aPos);
            aVInc.push_back(anInc);
            aVEcart.push_back(aEcart);

        }
        if (mExportPly)
            aPC.PutFile(StdPrefixGen(mGpsFile)+".ply");

        WriteImTmInFile(aOut,aVPos,aVInc,aVEcart,aAddFormat);
    }
}


int InterpImgPos_main(int argc,char ** argv)
{
    cIIP_Appli anAppli(argc,argv);
    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

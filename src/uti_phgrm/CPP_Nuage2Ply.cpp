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

#define DEF_OFSET -12349876


void Nuage2Ply_Banniere(string aNameOut)
{
	std::cout << " *********************************\n\n";
	std::cout << aNameOut << " exported \n\n";
	std::cout << " *********************************\n";
	std::cout << " *     Nuage (Cloud)             *\n";
	std::cout << " *     2 (to)                    *\n";
	std::cout << " *     Ply (.ply file)           *\n";
	std::cout << " *              Export DEM       *\n";
	std::cout << " *              (with texture)   *\n";
	std::cout << " *              as point cloud   *\n"; 
	std::cout << " *********************************\n\n";
}

int Ratio(double aV1,double aV2)
{

   double aRes = aV1 / aV2;
   int aIRes = round_ni(aRes);
   double anEcart =  (aV1 - aV2*aIRes)/aIRes;
   if ((ElAbs(anEcart) > 1.001) || (aIRes > 32))
      return -1;

   return aIRes;
}

int Nuage2Ply_main(int argc,char ** argv)
{
    std::string aNameNuage,aNameOut,anAttr1;
    std::vector<string> aVCom;
    int aBin  = 1;
    std::string aMask;
    double aSeuilMask=1;

    int DoPly = 1;
    int DoXYZ = 0;
    int DoNrm = 0;

    double aSc=1.0;
    double aDyn = 1.0;
    double aExagZ = 1.0;
    Pt2dr  aP0(0,0);
    Pt2dr  aSz(-1,-1);
    double aRatio = 1.0;
    bool aDoMesh = false;
    bool DoublePrec = false;
    Pt3dr anOffset(0,0,0);
    std::vector<Pt2dr> aBoxTerrain = {aP0,aP0,aP0};

    std::string  aNeighMask;
    int NormByC = 0;
    double DistCentre = 0;
    std::vector<string> aVNormName;
    bool ForceRGB=true;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aNameNuage,"Name of XML file", eSAM_IsExistFile),
    LArgMain()  << EAM(aSz,"Sz",true,"Sz (to crop)")
                    << EAM(aP0,"P0",true,"Origin (to crop)")
                    << EAM(aNameOut,"Out",true,"Name of output (default toto.xml => toto.ply)")
                    << EAM(aSc,"Scale",true,"Change the scale of result (def=1, 2 mean smaller)")
                    << EAM(anAttr1,"Attr",true,"Image to colour the point, or [R,G,B] when constant colour", eSAM_IsExistFile)
                    << EAM(aVCom,"Comments",true,"Commentary to add in the ply file (Def=None)", eSAM_NoInit )
                    << EAM(aBin,"Bin",true,"Generate Binary or Ascii (Def=1, Binary)")
                    << EAM(aMask,"Mask",true,"Supplementary mask image", eSAM_IsExistFile)
                    << EAM(aSeuilMask,"SeuilMask", true, "Theshold for supplementary mask")
                    << EAM(aDyn,"Dyn",true,"Dynamic of attribute")
                    << EAM(DoPly,"DoPly",true,"Do Ply, def = true")
                    << EAM(DoXYZ,"DoXYZ",true,"Do XYZ, export as RGB image where R=X,G=Y,B=Z")
                    << EAM(DoNrm,"Normale",true,"Add normale (Def=false, usable for Poisson)")
                    << EAM(NormByC,"NormByC",true,"Replace normal (Def=0, 2=optical center 1=point to center vector)",eSAM_InternalUse)
                    << EAM(DistCentre,"DistC",true,"Factor to set the distance of the camera (Altitude of pleiades : 694000)",eSAM_InternalUse)
                    << EAM(aVNormName,"NormName",true,"Replace normal name (Def=nx,ny,nz)", eSAM_NoInit )
                    << EAM(aExagZ,"ExagZ",true,"To exagerate the depth, Def=1.0")
                    << EAM(aRatio,"RatioAttrCarte",true,"")
                    << EAM(aDoMesh,"Mesh",true, "Do mesh (Def=false)")
                    << EAM(DoublePrec,"64B",true,"To generate 64 Bits ply, Def=false, WARN = do not work properly with meshlab or cloud compare")
                    << EAM(anOffset,"Offs", true, "Offset in points to limit 32 Bits accuracy problem")
                    << EAM(aNeighMask,"NeighMask",true,"Mask for neighboors when larger than point selection (for normals computation)")
                    << EAM(ForceRGB,"ForceRGB",true,"Force RGB even with gray image (Def=true because of bug in QT)")
                    << EAM(aBoxTerrain,"BoxTerrain",true,"Terrain coordinates within which the data should be exported [[xm,xM],[ym,yM],[zm,zM]]")
    );

    if (!MMVisualMode)
    {
    if (EAMIsInit(&aSz))
    {
         std::cout << "Waaaarnnn  :  meaning of parameter has changed\n";
         std::cout <<  " it used to be the corner (this was a bug)\n";
         std::cout <<  " now it is really the size\n";
    }

    if (aNameOut=="")
      aNameOut = StdPrefix(aNameNuage) + ".ply";

    cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromFileIm(aNameNuage,"XML_ParamNuage3DMaille","",aExagZ);

    if (aMask !="")
    {
         Im2D_Bits<1> aMaskN= aNuage->ImDef();
         Tiff_Im aMaskSup(aMask.c_str());
         ELISE_COPY(aMaskN.all_pts(),(aMaskSup.in(0) >= aSeuilMask) && aMaskN.in(), aMaskN.out());
    }

    if (aSz.x == -1 && aSz.y ==-1)
        aSz = Pt2dr(aNuage->SzUnique());

    bool ColVect = false;
    std::string aNameCoulTmp ;


    if ( ( anAttr1.length()!=0 ) && ( !ELISE_fp::exist_file( anAttr1 ) ) )
    {
        if ((anAttr1[0] == '[') && (anAttr1[anAttr1.length()-1] == ']'))
        {
            std::vector<int> aVCoul;
            FromString(aVCoul,anAttr1);
            if (aVCoul.size()==3)
            {
                ColVect = true;
                aNameCoulTmp = MMTemporaryDirectory() + "Tmp-CoulCstePly-"
                              +ToString(aVCoul[0])+"-"
                              +ToString(aVCoul[1])+"-"
                              +ToString(aVCoul[2])+".tif";
                Pt2di aSz = aNuage->SzUnique();
                bool Create;
                Tiff_Im aTF = Tiff_Im::CreateIfNeeded
                              (
                                  Create,
                                  aNameCoulTmp,
                                  aSz,
                                  GenIm::u_int1,
                                  Tiff_Im::No_Compr,
                                  Tiff_Im::RGB
                              );
                if (Create)
                {
                   ELISE_COPY
                   (
                       aTF.all_pts(),
                       Virgule(aVCoul[0],aVCoul[1],aVCoul[2]),
                       aTF.out()
                   );
                }
                anAttr1 = aNameCoulTmp;
            }
        }

        if (! ColVect)
        {
           cerr << "ERROR: colour image [" << anAttr1 << "] does not exist" << endl;
           return EXIT_FAILURE;
        }
    }



    if (anAttr1!="")
    {
       anAttr1 = NameFileStd(anAttr1,3,false,true,true,true);
       // std::cout << "ATTR1 " << anAttr1 << "\n";

       if (! EAMIsInit(&aRatio))
       {
            Pt2dr aSzNuage = Pt2dr(aNuage->SzUnique());
            Pt2dr aSzImage = Pt2dr(Tiff_Im(anAttr1.c_str()).sz());

            int aRx = Ratio(aSzImage.x,aSzNuage.x);
            int aRy = Ratio(aSzImage.y,aSzNuage.y);
            if ((aRx==aRy) && (aRx>0))
            {
               aRatio = aRx;
            }
            else
            {
               aRatio = 1;
               std::cout << "WARnnnnnnnnnnnn\n";
               std::cout << "Cannot get def value of RatioAttrCarte, set it to 1\n";
            }
            //std::cout << "RR " << aRx <<  " " << aRy << " SZss " << aSzNuage << aSzImage << "\n"; getchar();
       }
       aNuage->Std_AddAttrFromFile(anAttr1,aDyn,aRatio,ForceRGB);
    }

    // ATTENTION , SI &aNeighMask => IL FAUT QUE aRes soit egal a aNuage SANS passer par ReScaleAndClip
    cElNuage3DMaille * aRes = aNuage;

    //Application d'une box terrain (remplacement du crop pour que la recherche point par point ne soit pas trop longue)
    bool doBoxterrain = (aBoxTerrain[0].x != aBoxTerrain[0].y) && (aBoxTerrain[1].x != aBoxTerrain[1].y);
    if (doBoxterrain)
    {
        Pt3dr bt1(aBoxTerrain[0].x,aBoxTerrain[1].x,aBoxTerrain[2].x);
        Pt3dr bt2(aBoxTerrain[0].x,aBoxTerrain[1].x,aBoxTerrain[2].y);
        Pt3dr bt3(aBoxTerrain[0].x,aBoxTerrain[1].y,aBoxTerrain[2].x);
        Pt3dr bt4(aBoxTerrain[0].x,aBoxTerrain[1].y,aBoxTerrain[2].y);
        Pt3dr bt5(aBoxTerrain[0].y,aBoxTerrain[1].x,aBoxTerrain[2].x);
        Pt3dr bt6(aBoxTerrain[0].y,aBoxTerrain[1].x,aBoxTerrain[2].y);
        Pt3dr bt7(aBoxTerrain[0].y,aBoxTerrain[1].y,aBoxTerrain[2].x);
        Pt3dr bt8(aBoxTerrain[0].y,aBoxTerrain[1].y,aBoxTerrain[2].y);

        Pt2dr bi1 = aRes->Cam()->Ter2Capteur(bt1);
        Pt2dr bi2 = aRes->Cam()->Ter2Capteur(bt2);
        Pt2dr bi3 = aRes->Cam()->Ter2Capteur(bt3);
        Pt2dr bi4 = aRes->Cam()->Ter2Capteur(bt4);
        Pt2dr bi5 = aRes->Cam()->Ter2Capteur(bt5);
        Pt2dr bi6 = aRes->Cam()->Ter2Capteur(bt6);
        Pt2dr bi7 = aRes->Cam()->Ter2Capteur(bt7);
        Pt2dr bi8 = aRes->Cam()->Ter2Capteur(bt8);
        std::vector<double> vX = {bi1.x,bi2.x,bi3.x,bi4.x,bi5.x,bi6.x,bi7.x,bi8.x};
        std::vector<double> vY = {bi1.y,bi2.y,bi3.y,bi4.y,bi5.y,bi6.y,bi7.x,bi8.y};

        double xMin = *min_element(vX.begin(),vX.end());
        double xMax = *max_element(vX.begin(),vX.end());
        double yMin = *min_element(vY.begin(),vY.end());
        double yMax = *max_element(vY.begin(),vY.end());

        aP0=Pt2dr(xMin,yMin);
        aSz=Pt2dr(xMax-xMin,yMax-yMin);
    }

    if (EAMIsInit(&aNeighMask))
    {
        ELISE_ASSERT(   (aSc==1) && (aP0==Pt2dr(0,0)),"Can change scale && aNeighMask");
        Tiff_Im aTF(aNeighMask.c_str());
        Pt2di aSzN = aTF.sz();
        Im2D_Bits<1> aNM(aSzN.x,aSzN.y);
        ELISE_COPY(aNM.all_pts(),aTF.in()!=0,aNM.out());
        aRes->SetVoisImDef(aNM);
    }
    else
       aRes =  aNuage->ReScaleAndClip(Box2dr(aP0,aP0+aSz),aSc);
     //cElNuage3DMaille * aRes = aNuage;
    std::list<std::string > aLComment(aVCom.begin(), aVCom.end());
    if (NormByC)
    {
        if (! EAMIsInit(&DoNrm)) DoNrm = 5;
        aRes->SetNormByCenter(NormByC);
        aLComment.push_back("Norm is camera origin - NormByC : "+ToString(NormByC));
    }

    if (DistCentre && (NormByC==2))
    {
        aRes->SetDistCenter(DistCentre);
        aLComment.push_back("Center of camera set to a distance of : "+ToString(DistCentre));
    }

    if(doBoxterrain)
    {
        int nbPts = aRes->GetNbPts();
        for (Pt2di anI=aRes->Begin(); anI!=aRes->End() ;aRes->IncrIndex(anI))
        {
            Pt3dr aP = aRes->PtOfIndex(anI) - anOffset;
            bool xInfMin = (aP.x<min(aBoxTerrain[0].x,aBoxTerrain[0].y)- anOffset.x);
            bool xSupMax = (aP.x>max(aBoxTerrain[0].x,aBoxTerrain[0].y)- anOffset.x);
            bool yInfMin = (aP.y<min(aBoxTerrain[1].x,aBoxTerrain[1].y)- anOffset.y);
            bool ySupMax = (aP.y>max(aBoxTerrain[1].x,aBoxTerrain[1].y)- anOffset.y);

            if (xInfMin || xSupMax || yInfMin || ySupMax)
            {
                aRes->SetNoValue(anI);
                nbPts = nbPts-1;
                continue;
            }
        }
        aRes->SetNbPts(nbPts);
    }

    std::list<std::string > aLNormName(aVNormName.begin(), aVNormName.end());
    if (DoPly)
    {

       if (aDoMesh)
       {
           aRes->AddExportMesh();
       }

       aRes->PlyPutFile( aNameOut, aLComment, (aBin!=0), true, DoNrm, aLNormName, DoublePrec, anOffset);
    }
    if (DoXYZ)
    {
        aRes->NuageXZGCOL(StdPrefix(aNameNuage),DoublePrec);
    }

    cElWarning::ShowWarns(DirOfFile(aNameNuage)  + "WarnNuage2Ply.txt");
    if (ColVect)
    {
        // std::cout << "RRRRRRRrmmmmmmmmm Tmp Coul \n"; getchar();
        ELISE_fp::RmFile(aNameCoulTmp);
    }

	Nuage2Ply_Banniere(aNameOut);

    return EXIT_SUCCESS;

    }
    else return EXIT_SUCCESS;
}


int PlySphere_main(int argc,char ** argv)
{
    Pt3dr aC;
    Pt3di aCoul(255,0,0);
    double aRay;
    int aNbPts=5;
    std::string Out="Sphere.ply";


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aC,"Center of sphere")
                    << EAMC(aRay,"Ray of sphere"),
        LArgMain()  << EAM(aNbPts,"NbPts",true,"Number of Pts / direc (Def=5, give 1000 points)")
    );


    cPlyCloud aPC;
    aPC.AddSphere(aCoul,aC,aRay,aNbPts);
    aPC.PutFile(Out);

    return 1;
}

class cAppli_San2Ply_main : public cAppliWithSetImage
{
    public :
        cAppli_San2Ply_main(int argc,char ** argv);
};

cAppli_San2Ply_main::cAppli_San2Ply_main (int argc,char ** argv) :
    cAppliWithSetImage(argc-1,argv+1,0)
{
    std::string Out;
    double aDensity = 1.0;
    std::string aPat,anOri,aNameSan;


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aPat,"Pattern of image", eSAM_IsPatFile)
                    << EAMC(anOri ,"Orientation", eSAM_IsExistDirOri)
                    << EAMC(aNameSan,"Name of Analytical Surface", eSAM_IsExistFile),
        LArgMain()  << EAM(aDensity,"Density",true,"Factor proportional to point density")
                    << EAM(Out,"Out",true,"Name Of result")
    );

    if (MMVisualMode) return;

    if (!EAMIsInit(&Out)) Out = StdPrefix(aNameSan) + ".ply";

    std::cout << "NB IMAG " << mEASF.SetIm()->size() << "\n";

    cInterfSurfaceAnalytique * aSurf = cInterfSurfaceAnalytique::FromFile(aNameSan);

    cPlyCloud aPlyC;
    cParamISAPly aParam;

    aSurf->MakePly
    (
          aParam,
          aPlyC,
          VCam()
    );

    aPlyC.PutFile(Out);


/*
    cPlyCloud aPC;
    aPC.AddSphere(aCoul,aC,aRay,aNbPts);
    aPC.PutFile(Out);

    return 1;
*/
}
/*
*/

int San2Ply_main(int argc,char ** argv)
{
    cAppli_San2Ply_main anAppli(argc,argv);
    return EXIT_SUCCESS;
}





int PlyGCP_main(int argc,char ** argv)
{
    cPlyCloud aPC;
    Pt3di aCoul(255,0,0);
    std::string aNameGCP,aNamePly;
    Pt3dr aNorm (0,0,1);
    double aResol;
    double aOffset = 2.0;
    bool aUseNum = false;
    int aNbByCase = 5;
    double aSpace = 0.0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameGCP,"Name of GCP  file", eSAM_IsExistFile)
                    << EAMC(aResol,"Resolution, use for string"),
        LArgMain()  << EAM(aNamePly,"Out",true," Def= GCP.ply")
                    << EAM(aNorm,"Normal",true,"Def=(0,0,1)")
                    << EAM(aCoul,"Coul",true,"Color Def=[255,0,0]")
                    << EAM(aOffset,"Offset",true, "Ofset, prop to Resolution, Def=2")
                    << EAM(aUseNum,"UseNum",true, "Use num as name, def=false")
                    
    );
    if (!EAMIsInit(&aNamePly)) 
    {
         aNamePly = StdPrefix(aNameGCP) + ".ply";
    }
    Pt3dr aX,aY;
    MakeRONWith1Vect(aNorm,aX,aY);

    cDicoAppuisFlottant aDAF = StdGetFromPCP(aNameGCP,DicoAppuisFlottant);

    int aNum= 1;
    for (std::list<cOneAppuisDAF>::const_iterator itF=aDAF.OneAppuisDAF().begin() ; itF!=aDAF.OneAppuisDAF().end() ; itF++)
    {
        cOneAppuisDAF anAp = *itF;
        std::string aStr = anAp.NamePt();
        if (aUseNum) aStr = ToString(aNum);
        Pt3dr aP0 = anAp.Pt() + aNorm * (aOffset*aResol);

        aPC.PutStringDigit(aStr,aP0,aX,-aY,aCoul,aResol,aResol*aSpace,aNbByCase);

        aNum++;
    }

    aPC.PutFile(aNamePly);

    return EXIT_SUCCESS;
}


/**************************************************************/
/*                                                            */
/*             cApply_PlyCamOrthoC                            */
/*                                                            */
/**************************************************************/

class cApply_PlyCamOrthoC
{
     public :
         cApply_PlyCamOrthoC(int argc,char ** argv);
     private :

         int  mNbPoints;
         int  mNbTeta;
         std::vector<Pt3dr> mPts;
         Pt3dr mDirCam1;
         Pt3dr mDirCam2;
         Pt3dr mNorm;
         double mDist;

         void MakePlySec(double aTeta,const std::string & aName,Pt3di aCoul);
         void AddFaisceauMaster(std::vector<ElSeg3D> & aRes,Pt3dr  aDir,const std::string & aName,Pt3di aCou);
         void MakePly(const std::vector<ElSeg3D> & aVS,const std::string & aName,Pt3di aCoul);

         vector<ElSeg3D> mMast;
         vector<ElSeg3D> mSec1;
};

void cApply_PlyCamOrthoC::MakePlySec(double aTeta,const std::string & aName,Pt3di aCoul)
{
   ElMatrix<REAL>  aR =  VectRotationArroundAxe(mNorm,aTeta);

   vector<ElSeg3D> aNew;
   for (int aK=0 ; aK<int(mSec1.size()) ; aK++)
   {
         aNew.push_back(ElSeg3D(aR*mSec1[aK].P0(),aR*mSec1[aK].P1()));
   }
   MakePly(aNew,aName,aCoul);
}



void cApply_PlyCamOrthoC::MakePly(const std::vector<ElSeg3D> & aVS,const std::string & aName,Pt3di aCoul)
{
   cPlyCloud aPC;

   for (int aK=0 ; aK<int(aVS.size()) ; aK++)
   {
       aPC.AddSeg(aCoul,aVS[aK].P0(),aVS[aK].P1(),200);
       aPC.AddSphere(aCoul,aVS[aK].P0(),0.1,5);
   }
   
   aPC.PutFile(aName);
}

void cApply_PlyCamOrthoC::AddFaisceauMaster(std::vector<ElSeg3D> & aRes,Pt3dr  aDir,const std::string & aName,Pt3di aCoul)
{
    aDir = vunit(aDir);

    ElSeg3D aSeg(Pt3dr(0,0,0),aDir);

    for (int aK=0 ; aK<mNbPoints ; aK++)
    {
          Pt3dr aProj = mPts[aK]-aSeg.ProjOrtho(mPts[aK]); // Proj sur le plan ortho

          aRes.push_back(ElSeg3D(aProj-aDir*mDist,aProj+aDir*mDist));
    }
    MakePly(aRes,aName,aCoul);
}

cApply_PlyCamOrthoC::cApply_PlyCamOrthoC(int argc,char ** argv)  :
    mNbPoints (6),
    mNbTeta   (10),
    mDirCam1  (vunit(Pt3dr(1,0,0))),
    mDirCam2  (vunit(Pt3dr(1,1,0))),
    mNorm     (mDirCam1^mDirCam2),
    mDist     (5.0)
{
    cPlyCloud aPC;
     for (int aK=0 ; aK< mNbPoints ; aK++)
     {
          mPts.push_back(Pt3dr(NRrandC(),NRrandC(),NRrandC()));
          aPC.AddSphere(Pt3di(0,255,0),mPts.back(),0.02,5);
     }
     aPC.PutFile("Inter12.ply");

     AddFaisceauMaster(mMast,mDirCam1,"Cam1.ply",Pt3di(255,0,0));
     AddFaisceauMaster(mSec1,mDirCam2,"Cam2.ply",Pt3di(0,0,255));

     MakePlySec(0.4,"Cam3.ply",Pt3di(0,128,255));
     MakePlySec(1.7,"Cam4.ply",Pt3di(128,0,255));

     cPlyCloud aPC2;
     aPC2.AddSeg(Pt3di(0,255,0),mNorm*mDist*2,mNorm*mDist*-2,4000);
     aPC2.PutFile("Axe.ply");
}

int MakePly_CamOrthoC(int argc,char ** argv)
{
    cApply_PlyCamOrthoC anAply(argc,argv);
    return EXIT_SUCCESS;
}








/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

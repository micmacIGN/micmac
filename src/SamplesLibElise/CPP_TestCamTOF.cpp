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


class cAppliCamTOF
{
    public : 
        cAppliCamTOF(int argc,char** argv);

    private : 

        void        SetResolPlani(const std::vector<Pt3dr>&);
       
        bool        WriteNuage    (); 
        bool        WritePCDToTIF ();
        bool        ParsePCDHeader(std::string&);
        bool        ParsePCDXYZ   ();
        void        GetStr        (char*& aPtr,std::string*);
        void        GetStrVec     (char*& aPtr,std::vector<std::string>&);
        std::string CreateName    (int);

        Pt2dr       mResolPlani;
        double      mResolPlaniMoy;
        Pt2dr       mResolPlaniSig;
        Pt3dr       mPP;
        Pt3dr       mCG; //center of gravity
        int         mNumPt; 
        Pt2di       mSz;
        
        std::string mName;    
        std::string mOut;    
};

cAppliCamTOF::cAppliCamTOF(int argc,char** argv) :
    mPP(Pt3dr(0,0,0)),
    mCG(Pt3dr(0,0,0))
{

    bool DoPly=false;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mName,"PCD file"),
        LArgMain() << EAM(mOut,"Out",true, "Name of output file")
                   << EAM(DoPly,"Ply",true, "Create ply, Def=false")
    );

    
    if (!WritePCDToTIF())
        ELISE_ASSERT(false,"cAppliCamTOF::cAppliCamTOF  , cannot read pcd file");

    if (!WriteNuage())
        ELISE_ASSERT(false,"cAppliCamTOF::cAppliCamTOF  , cannot save the Nuage");

    if (DoPly)
    {
        if (ELISE_fp::exist_file(CreateName(3)))
        {
            int DoNrm=0;
            std::list<std::string> aLC;
            std::list<std::string> aLNN;
            cElNuage3DMaille* aNuage = cElNuage3DMaille::FromFileIm(CreateName(3),"XML_ParamNuage3DMaille","",1.0);
            aNuage->PlyPutFile(CreateName(4),aLC, true, true, DoNrm, aLNN, true);
        }
        else
            ELISE_ASSERT(false,"cAppliCamTOF::cAppliCamTOF NuageProf.xml file missing");
    }

    std::cout << "Saved files: " << CreateName(0) << ", " << CreateName(1) << ", " << CreateName(2)
                                 << ", " << CreateName(3) ;
    if (DoPly)
        std::cout << ", " << CreateName(4) << "\n";
    else
        std::cout << "\n";
    

    BanniereMM3D();


}

std::string cAppliCamTOF::CreateName(int aK)
{
    std::string aRes;

    switch (aK)
    {
        case 0 : 
            aRes = mOut + ".tif";
            break;
        case 1 :
            aRes = mOut + "_Prof.tif";
            break;
        case 2 :
            aRes = mOut + "_Masq.tif";
            break;
        case 3 : 
            aRes = mOut+"_NuageProf.xml"; 
            break;
        case 4 : 
            aRes = mOut+"_NuageProf.ply"; 
            break;

    }

    return aRes; 
}

void cAppliCamTOF::GetStrVec(char*& aPtr,std::vector<std::string>& aStrVec)
{

    while ((*aPtr) != '\0')
    {
        
        std::string * aPStr = new std::string();


        GetStr(aPtr,aPStr);
        aStrVec.push_back(*aPStr);
        
        aPtr++;

        delete aPStr;

    }


}
void cAppliCamTOF::GetStr(char*& aPtr,std::string* aRes)
{

    while (((*aPtr)!=' ') && ((*aPtr) != '\0') )
    {
        (*aRes) += (*aPtr);

        aPtr++;
    }
}

void cAppliCamTOF::SetResolPlani(const std::vector<Pt3dr>& aN)
{
    int aK1=0,aK2=0;
    int aDx=0,aDy=0;

    for (int aK=1; aK<int(aN.size()); aK++)
    {
        if ((aK1+2)==mSz.x)
        {
            double aDiff = (aN.at(aK).x-aN.at(aK-mSz.x+1).x);
            mResolPlani.x    += aDiff;
            mResolPlaniSig.x += ElSquare(aDiff);
            aDx++;
        }  

        if ((aK2+2)==mSz.y)
        {
            double aDiff = aN.at(aK).y-aN.at(aK-aK2*mSz.x-aK1).y;
            mResolPlani.y    += aDiff;
            mResolPlaniSig.y += ElSquare(aDiff);
            aDy++;
        } 

        //increment K1,K2
        (aK1+1) >= mSz.x ? aK1=0,aK2++ : aK1++;        
    }   
    
    /* Resolution calculated per line */ 
    mResolPlani.x /= aDx; 
    mResolPlani.y /= aDy;
 
    mResolPlaniSig.x /= aDx; 
    mResolPlaniSig.y /= aDy;
    
    mResolPlaniSig.x -= ElSquare(mResolPlani.x); 
    mResolPlaniSig.y -= ElSquare(mResolPlani.y); 
    
    /* Resolution calculated per pixel */
    mResolPlani.x /= mSz.x; 
    mResolPlani.y /= mSz.y;
    
    mResolPlaniSig.x /= mSz.x;    
    mResolPlaniSig.y /= mSz.y;    

    mResolPlaniMoy = 0.5*(double(1)/mResolPlani.x + double(1)/mResolPlani.y);

    std::cout << "Moy ResolPlani=" << mResolPlani << ", sigma=" << mResolPlaniSig << "\n";

 

}

bool cAppliCamTOF::ParsePCDXYZ()
{
    Im2D_REAL4 aImProf(mSz.x,mSz.y);
    Im2D_REAL4 aIm(mSz.x,mSz.y);

    std::vector<Pt3dr>   aNuage;

    ELISE_fp aFIn(mName.c_str(),ELISE_fp::READ);
    char * aLine;

    int aK1=0,aK2=0;
    std::vector<std::string> aPStrVec;

    //iterate to XYZ
    for (int aK=0; aK<11; aK++)
        aFIn.std_fgets();

    while ((aLine = aFIn.std_fgets()))
    {
        char * it = aLine;
        std::string aPStr;
        std::vector<std::string> aPStrVec;
        GetStrVec(it,aPStrVec);
       
        aImProf.SetR(Pt2di(aK1,aK2),std::atof(aPStrVec.at(2).c_str()));
        aIm.SetR(Pt2di(aK1,aK2),std::atof(aPStrVec.at(3).c_str()));

        aNuage.push_back(Pt3dr(std::atof(aPStrVec.at(0).c_str()),
                               std::atof(aPStrVec.at(1).c_str()),
                               std::atof(aPStrVec.at(2).c_str())));


        //center of gravity
        mCG.x += std::atof(aPStrVec.at(0).c_str());
        mCG.y += std::atof(aPStrVec.at(1).c_str());
        mCG.z += std::atof(aPStrVec.at(2).c_str());

        //increment K1,K2
        (aK1+1) >= mSz.x ? aK1=0,aK2++ : aK1++;        

    }
    aFIn.close();
    mCG.x /= mNumPt;
    mCG.y /= mNumPt;
    mCG.z /= mNumPt;

    //my hypothesis : 1st pt in cloud is 1st pt in sensor geometry 
    mPP = aNuage.at(0); 

    SetResolPlani(aNuage);

    if (!EAMIsInit(&mOut))
        mOut = StdPrefix(mName);

    std::string aImName = CreateName(0);
    std::string aImPName = CreateName(1);
    std::string aImMName = CreateName(2);
    
    /* Intensity image */
    ELISE_COPY
    (
        aIm.all_pts(),
        aIm.in(),
        Tiff_Im(
            aImName.c_str(),
            mSz,
            GenIm::real4,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero,
            Tiff_Im::Empty_ARG ).out()
    );

    /* Depth map */
    ELISE_COPY
    (
        aImProf.all_pts(),
        aImProf.in(),
        Tiff_Im(
            aImPName.c_str(),
            mSz,
            GenIm::real4,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero,
            Tiff_Im::Empty_ARG ).out()
    );

    /* Masq */
    ELISE_COPY
    (
        aIm.all_pts(),
        1,
        Tiff_Im(
            aImMName.c_str(),
            mSz,
            GenIm::bits1_msbf,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero,
            Tiff_Im::Empty_ARG ).out()
    );


    return true;    
}


bool cAppliCamTOF::ParsePCDHeader(std::string& aStr)
{
    ELISE_fp aFIn(mName.c_str(),ELISE_fp::READ);
    char * aLine;

    std::vector<std::string> aPStrVec;
    while ((aLine = aFIn.std_fgets()))
    {
        char * it = aLine;
        std::vector<std::string> aPStrVec;
        
        GetStrVec(it,aPStrVec);
        
        if (aPStrVec.size() > 1)
            if (aPStrVec.at(0).compare(aStr) == 0)
            {
                mNumPt = std::atof(aPStrVec.at(1).c_str());
                if ((mNumPt/240) == 320)               
                {
                    mSz = Pt2di(320,240);
                    aFIn.close();

                    return true;
                }
                else
                {
                    aFIn.close();
                    ELISE_ASSERT(false, "cAppliCamTOF::ParsePCDHeader not supported sensor format"); 
                }
            }
    }
    
    aFIn.close();
    
    return false;
}

bool cAppliCamTOF::WriteNuage()
{

    cXML_ParamNuage3DMaille aNuageXML;

 
    cPN3M_Nuage            aPN3M;
    cImage_Profondeur      aImProf;
    aImProf.Image()          = CreateName(1);
    aImProf.Masq()           = CreateName(2);
    aImProf.OrigineAlti()    = 0;
    aImProf.ResolutionAlti() = 1;//0.5*(double(1)/mResolPlani.x + double(1)/mResolPlani.y);//1;
    aImProf.GeomRestit()     = eGeomMNTFaisceauIm1PrCh_Px1D;
    //aImProf.GeomRestit()     = eGeomMNTEuclid;
    aPN3M.Image_Profondeur() = aImProf;

    aNuageXML.PN3M_Nuage()        = aPN3M;
    
    cPM3D_ParamSpecifs        aParSpec;
    cModeFaisceauxImage       aMFI;
    aMFI.DirFaisceaux()           = Pt3dr(0,0,1);
    aMFI.ZIsInverse()             = false;
    aParSpec.ModeFaisceauxImage() = aMFI;
    aNuageXML.PM3D_ParamSpecifs() = aParSpec;  
    
    aNuageXML.NbPixel()           = mSz;  
    aNuageXML.SsResolRef()        = 1;  

    cOrientationConique       aOC;
    cOrientationExterneRigide aOER;

    cRotationVect             aRotV;
    cTypeCodageMatr           aCMat;

    aCMat.L1() = Pt3dr(1,0,0);
    aCMat.L2() = Pt3dr(0,1,0);
    aCMat.L3() = Pt3dr(0,0,1);
    
    aRotV.CodageMatr() = aCMat;
 
    aOER.ParamRotation()      = aRotV;
    //aOER.Centre()             = Pt3dr(0,0,-1);
    aOER.Centre()             = Pt3dr(mPP.x,mPP.y,-1);
    aOC.Externe()             = aOER;
    aOC.ConvOri().KnownConv() = eConvApero_DistM2C; 
    aOC.TypeProj()            = eProjOrthographique;
   std::cout << "eeeeee centre=" << mPP << "\n";  
    cAffinitePlane  aAffP;
    aAffP.I00()               = Pt2dr(0,0);
    aAffP.V10()               = Pt2dr(mResolPlaniMoy,0);
    aAffP.V01()               = Pt2dr(0,mResolPlaniMoy);
    aOC.OrIntImaM2C()         = aAffP;    

 
    cCalibrationInternConique aCIC;
    cCalibDistortion  aCD;
    cModNoDist aNoDist;
    aCD.ModNoDist().SetVal(aNoDist);
    aOC.Interne() = aCIC;
    aOC.Interne().Val().F()   = 0;
    aOC.Interne().Val().PP() = Pt2dr(12345678,87654321);
    aOC.Interne().Val().SzIm() = mSz;
    aOC.Interne().Val().CalibDistortion().push_back(aCD);
    
    aNuageXML.Orientation()   = aOC;
    
    //aNuageXML.RatioResolAltiPlani() = 0.5*(double(1)/mResolPlani.x + double(1)/mResolPlani.y);

    MakeFileXML(aNuageXML,CreateName(3)); 

    return true;
}

bool cAppliCamTOF::WritePCDToTIF()
{
  
    std::string aStr = "WIDTH"; 
    if (!ParsePCDHeader(aStr))
        return false;

    if (!ParsePCDXYZ())
        return false;
 
    return true;
}


int TestCamTOF_main(int argc,char** argv)
{
    cAppliCamTOF aAppTOF(argc,argv);

    return 0;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

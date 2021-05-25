/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de           Correlation
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
#include "../uti_phgrm/Apero/BundleGen.h"

//visualize satellite 2D image deformation
int CPP_SATDef2D_main(int argc,char ** argv)
{
    std::string aGBOriName = "";
    std::string aNameOut = "_D2D.tif";
    std::string aDir = "";
    std::string aFile = "";

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aGBOriName,"Corrected image orientation file (type Xml_CamGenPolBundle)"),
        LArgMain() << EAM(aNameOut,"Out",true,"Output filename")
    );
    
    cPolynomial_BGC3M2D * aCam = cPolynomial_BGC3M2D::NewFromFile(aGBOriName);

    SplitDirAndFile(aDir, aFile, aGBOriName);
    
    int aStep = 100;
    Pt2di aSzOrg = Pt2di(2*aCam->Center().x,2*aCam->Center().y);
    Pt2di aSzSca = Pt2di(aSzOrg.x/aStep, aSzOrg.y/aStep);
    //float aScale = (float) aSzSca.x / aSzOrg.x;

    GenIm::type_el aTypeOut = GenIm::u_int1;
    Tiff_Im::COMPR_TYPE aModeCompr = Tiff_Im::No_Compr;

    REAL GS1 = 0;
    Disc_Pal aPP1 =  Disc_Pal::PCirc(256);
    Elise_colour * Cols = aPP1.create_tab_c();
    Cols[0] = Elise_colour::gray(GS1);
    Disc_Pal aPal (Cols,256);

    L_Arg_Opt_Tiff aLArgTiff = Tiff_Im::Empty_ARG;

    Tiff_Im aTiffX  = Tiff_Im
                       (
                           (aDir + "/x" + aNameOut).c_str(),
                           aSzSca,
                           aTypeOut,
                           aModeCompr,
                           Tiff_Im::BlackIsZero,//aPal
                           aLArgTiff
                       );

    Tiff_Im aTiffY  = Tiff_Im
                       (
                           (aDir + "/y" + aNameOut).c_str(),
                           aSzSca,
                           aTypeOut,
                           aModeCompr,
                           Tiff_Im::BlackIsZero,//aPal,
                           aLArgTiff
                       );

    Tiff_Im aTiffXY  = Tiff_Im
                       (
                           (aDir + "/xy" + aNameOut).c_str(),
                           aSzSca,
                           aTypeOut,
                           aModeCompr,
                           Tiff_Im::BlackIsZero,//aPal,
                           aLArgTiff
                       );

    Fonc_Num aResX, aResY, aResXY;
    double aGMinx, aGMaxx,aGMiny, aGMaxy, aGMinxy, aGMaxxy;   
    int aP1, aP2;
    Pt2dr aPD;
    double aDSum;

    //initialize
    aPD = aCam->DeltaCamInit2CurIm(Pt2dr(0,0));
    aDSum = std::sqrt(pow(aPD.x,2) + pow(aPD.y,2));
    aGMinx=aPD.x;
    aGMaxx=aPD.x;
    aGMiny=aPD.y;
    aGMaxy=aPD.y;
    aGMinxy=aDSum;
    aGMaxxy=aDSum; 
    
    TIm2D<REAL,REAL> aImX(aSzSca), aImY(aSzSca), aImXY(aSzSca);

    for(aP1=0; aP1<aSzSca.x; aP1++)
    {
        for(aP2=0; aP2<aSzSca.y; aP2++)
        {


            aPD = aCam->DeltaCamInit2CurIm(Pt2dr(aP1*aStep,aP2*aStep));
            aDSum = std::sqrt(pow(aPD.x,2) + pow(aPD.y,2));

            aImX.oset(Pt2di(aP1,aP2),aPD.x);
            aImY.oset(Pt2di(aP1,aP2),aPD.y);
            aImXY.oset(Pt2di(aP1,aP2),aDSum);
    

            if( aGMinx>aPD.x )
                aGMinx=aPD.x;

            if( aGMaxx<aPD.x )
                aGMaxx=aPD.x;

            if( aGMiny>aPD.y )
                aGMiny=aPD.y;

            if( aGMaxy<aPD.y )
                aGMaxy=aPD.y;
            
            if( aGMinxy>aDSum )
                aGMinxy=aDSum;

            if( aGMaxxy<aDSum )
                aGMaxxy=aDSum;

            // std::cout << " " << aPD; 
        }
        //std::cout << "\n"; 
    }


    
    //in case of flat displacements
    if(aGMinx < 0)
    {
        if(aGMinx == aGMaxx)
            aGMaxx=0; 
    }
    else        
    { 
        if(aGMinx == aGMaxx)  
            aGMinx=0; 
    }
    if(aGMiny < 0)
    { 
        if(aGMiny == aGMaxy)  
            aGMaxy=0; 
    }
    else        
    { 
        if(aGMiny == aGMaxy) 
            aGMiny=0; 
    }
    if(aGMinxy < 0)
    { 
        if(aGMinxy == aGMaxxy) 
            aGMaxxy=0; 
    }
    else        
    { 
        if(aGMinxy == aGMaxxy) 
            aGMinxy=0; 
    }

    std::cout.precision(15);
    std::cout << "displacement in x:  GMin,Gax " << aGMinx << " " << aGMaxx << "\n";
    std::cout << "displacement in y:  GMin,Gax " << aGMiny << " " << aGMaxy << "\n";
    std::cout << "displacement in xy:  GMin,Gax " << aGMinxy << " " << aGMaxxy << "\n";



    aResX = (aImX.in() - aGMinx) * (255.0 / ElMax(aGMaxx-aGMinx,1e-3));
    //aResX = StdFoncChScale(aImX,Pt2dr(0,0), Pt2dr(1.f/aScale,1.f/aScale));

    ELISE_COPY
    (
        aTiffX.all_pts(),
        aResX,
        aTiffX.out()
    );

    

    aResY = (aImY.in() - aGMiny) * (255.0 / ElMax(aGMaxy-aGMiny,1e-3));
    //aResY = StdFoncChScale(aImY,Pt2dr(0,0), Pt2dr(1.f/aScale,1.f/aScale));

    ELISE_COPY
    (
        aTiffY.all_pts(),
        aResY,
        aTiffY.out()
    );



    aResXY = (aImXY.in() - aGMinxy) * (255.0 / ElMax(aGMaxxy-aGMinxy,1e-3));

    ELISE_COPY
    (
        aTiffXY.all_pts(),
        aResXY,
        aTiffXY.out()
    );

    return EXIT_SUCCESS;
}

//wrong! coords not normalized! visualize satellite 2D image deformation
int CPP_SATDef2D_main0(int argc,char ** argv)
{
    std::string aGBOriName = "";
    std::string aNameOut = "_D2D.tif";

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aGBOriName,"Corrected image orientation file (type   Xml_CamGenPolBundle)"),
        LArgMain() << EAM(aNameOut,"Out",true,"Output filename")
    );

    cXml_CamGenPolBundle aXml = StdGetFromSI(aGBOriName,Xml_CamGenPolBundle);
    
    int aStep = 100;
    Pt2di aSzOrg = Pt2di(2*aXml.Center().x,2*aXml.Center().y);
    Pt2di aSzSca = Pt2di(aSzOrg.x/aStep, aSzOrg.y/aStep);
    //float aScale = (float) aSzSca.x / aSzOrg.x;


    GenIm::type_el aTypeOut = GenIm::u_int1;
    Tiff_Im::COMPR_TYPE aModeCompr = Tiff_Im::No_Compr;

    REAL GS1 = 0;
    Disc_Pal aPP1 =  Disc_Pal::PCirc(256);
    Elise_colour * Cols = aPP1.create_tab_c();
    Cols[0] = Elise_colour::gray(GS1);
    Disc_Pal aPal (Cols,256);
    Gray_Pal Pgr (30);

    L_Arg_Opt_Tiff aLArgTiff = Tiff_Im::Empty_ARG;

    Tiff_Im aTiffX  = Tiff_Im
                       (
                           ("X" + aNameOut).c_str(),
                           aSzSca,
                           aTypeOut,
                           aModeCompr,
                           aPal,//Tiff_Im::BlackIsZero,
                           aLArgTiff
                       );

    Tiff_Im aTiffY  = Tiff_Im
                       (
                           ("Y" + aNameOut).c_str(),
                           aSzSca,
                           aTypeOut,
                           aModeCompr,
                           aPal,
                           aLArgTiff
                       );

    Tiff_Im aTiffXY  = Tiff_Im
                       (
                           ("XY" + aNameOut).c_str(),
                           aSzSca,
                           aTypeOut,
                           aModeCompr,
                           aPal,
                           aLArgTiff
                       );

    Fonc_Num aResX, aResY, aResXY;
    unsigned int aK;
    int aP1, aP2;

//ToPNorm(const Pt2dr aP) const {return (aP-mCenter)/mAmpl;}
//FromPNorm(const Pt2dr aP) const {return aP*mAmpl + mCenter;}

    TIm2D<INT,INT> aImX(aSzSca), aImY(aSzSca), aImXY(aSzSca);

    for(aP1=0; aP1<aSzSca.x; aP1++)
    {
        for(aP2=0; aP2<aSzSca.y; aP2++)
        {

            double aTx=0, aTy=0;

            for(aK=0; aK<aXml.CorX().Monomes().size(); aK++)
            {

                if(aXml.CorX().Monomes()[aK].mDegX==0 &&
                   aXml.CorX().Monomes()[aK].mDegY==0)
                {
                   aTx += aXml.CorX().Monomes()[aK].mCoeff;
                }
                else if(aXml.CorX().Monomes()[aK].mDegX==0)
                {
                   aTx += aXml.CorX().Monomes()[aK].mCoeff*
                         pow(double(aP2*aStep),double(aXml.CorX().Monomes()[aK].mDegY));
                }
                else if(aXml.CorX().Monomes()[aK].mDegY==0)
                {
                   aTx += aXml.CorX().Monomes()[aK].mCoeff*
                         pow(double(aP1*aStep),double(aXml.CorX().Monomes()[aK].mDegX));
                }
                else
                {
                   aTx += aXml.CorX().Monomes()[aK].mCoeff*
                         pow(double(aP1*aStep),double(aXml.CorX().Monomes()[aK].mDegX))*
                         pow(double(aP2*aStep),double(aXml.CorX().Monomes()[aK].mDegY));
                }
            }


            for(aK=0; aK<aXml.CorY().Monomes().size(); aK++)
            {


                if(aXml.CorY().Monomes()[aK].mDegX==0 &&
                   aXml.CorY().Monomes()[aK].mDegY==0)
                {
                   aTy += aXml.CorY().Monomes()[aK].mCoeff;
                }
                else if(aXml.CorY().Monomes()[aK].mDegX==0)
                {
                    aTy += aXml.CorY().Monomes()[aK].mCoeff*
                           pow(double(aP2*aStep),double(aXml.CorY().Monomes()[aK].mDegY));
                }
                else if(aXml.CorY().Monomes()[aK].mDegY==0)
                {
                    aTy += aXml.CorY().Monomes()[aK].mCoeff*
                           pow(double(aP1*aStep),double(aXml.CorX().Monomes()[aK].mDegX));
                }
                else
                {
                    aTy += aXml.CorY().Monomes()[aK].mCoeff*
                           pow(double(aP1*aStep),double(aXml.CorY().Monomes()[aK].mDegX))*
                           pow(double(aP2*aStep),double(aXml.CorY().Monomes()[aK].mDegY));
                }
            }


            aImX.oset(Pt2di(aP1,aP2),aTx);
            aImY.oset(Pt2di(aP1,aP2),aTy);
            aImXY.oset(Pt2di(aP1,aP2),abs(aTx)+abs(aTy));
//            std::cout << aP1 << " " << aP2 << " =" << aTy << "\n"; 

        }
    }
    REAL GMin=0,GMax=0;
    ELISE_COPY
    (
        aImX.all_pts(),
        aImX.in(),
        VMax(GMax)|VMin(GMin)
    );


    //in case of flat displacements
    if(GMin < 0){ GMin == GMax ? GMax=0 : GMax=GMax; }
    else        { GMin == GMax ? GMin=0 : GMin=GMin; }

    std::cout << "displacement in x:  GMin,Gax " << GMin << " " << GMax << "\n";

    aResX = (aImX.in() - GMin) * (255.0 / (GMax-GMin));
    //aResX = StdFoncChScale(aImX,Pt2dr(0,0), Pt2dr(1.f/aScale,1.f/aScale));

    ELISE_COPY
    (
        aTiffX.all_pts(),
        aResX,
        aTiffX.out()
    );

    GMin=0;
    GMax=0;
    ELISE_COPY
    (
        aImY.all_pts(),
        aImY.in(),
        VMax(GMax)|VMin(GMin)
    );

    //in case of flat displacements 
    if(GMin < 0){ GMin == GMax ? GMax=0 : GMax=GMax; }
    else        { GMin == GMax ? GMin=0 : GMin=GMin; }
    std::cout << "displacement in y:  GMin,Gax " << GMin << " " << GMax << "\n";

    aResY = (aImY.in() - GMin) * (255.0 / (GMax-GMin));
    //aResY = StdFoncChScale(aImY,Pt2dr(0,0), Pt2dr(1.f/aScale,1.f/aScale));

    ELISE_COPY
    (
        aTiffY.all_pts(),
        aResY,
        aTiffY.out()
    );

    GMin=0;
    GMax=0;
    ELISE_COPY
    (
        aImXY.all_pts(),
        aImXY.in(),
        VMax(GMax)|VMin(GMin)
    );

    //in case of flat displacements 
    if(GMin < 0){ GMin == GMax ? GMax=0 : GMax=GMax; }
    else        { GMin == GMax ? GMin=0 : GMin=GMin; }
    std::cout << "displacement in xy: GMin,Gax " << GMin << " " << GMax << "\n";

    aResXY = (aImXY.in() - GMin) * (255.0 / (GMax-GMin));

    ELISE_COPY
    (
        aTiffXY.all_pts(),
        aResXY,
        aTiffXY.out()
    );

    return EXIT_SUCCESS;
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


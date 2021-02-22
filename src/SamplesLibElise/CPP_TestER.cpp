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
#include "../uti_phgrm/Apero/cCameraRPC.h"
#include "../uti_phgrm/NewOri/NewOri.h"
#include "general/ptxd.h"
#include "../../include/im_tpl/cPtOfCorrel.h"
#include "../uti_phgrm/TiepTri/MultTieP.h"

void CheckBounds(Pt2dr & aPmin, Pt2dr & aPmax, const Pt2dr & aP, bool & IS_INI);

class cAppliTuak
{
    public:
        cAppliTuak(int argc,char ** argv);

    private:
        std::string mDir;
        std::string mIms;
        std::string mOri;


};

cAppliTuak::cAppliTuak(int argc,char ** argv):
    mDir(""),
    mIms(""),
    mOri("")
{
    std::string aFullName = "";
    std::string aComMalt = "";
    
    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aFullName,"Full Name (Dir+Pattern)")
                   << EAMC(mOri,"Orientation directory"),
        LArgMain()
    );

    SplitDirAndFile(mDir,mIms,aFullName);
    setInputDirectory(mDir);

    aComMalt = MM3dBinFile_quotes("Malt")
           + " UrbanMNE " 
           + aFullName + " "
           + mOri +
           + " NbVI=2 NbProc=1";

    TopSystem(aComMalt.c_str());
}


//salient points
int TestER_Tuak_main(int argc,char ** argv)
{
    cAppliTuak aAppli(argc,argv);

    //tile the image
    //run matching without regularization on respective tiles 
    //       in eGeomMNTFaisceauIm1ZTerrain Px1D 
    //       (result saved to hard drive)
    //read the correlations and depth fields
    //given the connectivity take the left image + its correlation image
    //       pick the salient points (corrlation works as a mask)
    //       get the depth for every point and backproject to all connected imgs
    //       you can do poly fitting on the coeff to get subpix accuracy
    //       save the points just as they are saved in Tapas (same data structure)

    //* cAppliKeyPtsPB?
    //if I give Malt many images..how does it pick the matching order...intercept this info to your appli

    return EXIT_SUCCESS;
}

//visualize satellite image deformation
int TestER_main(int argc,char ** argv)
{
    std::string aGBOriName = "";
    std::string aNameOut = "2Deform.tif";

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aGBOriName,"Corrected image orientation file (type Xml_CamGenPolBundle)"),
        LArgMain() << EAM(aNameOut,"Out",true,"Output filename")	
    );

    cXml_CamGenPolBundle aXml = StdGetFromSI(aGBOriName,Xml_CamGenPolBundle);
    Pt2di aSzOrg = Pt2di(2*aXml.Center().x,2*aXml.Center().y);
    Pt2di aSzSca = Pt2di(aSzOrg.x/100, aSzOrg.y/100); 
    //float aScale = (float) aSzSca.x / aSzOrg.x;
    int aStep = (int) aSzOrg.x / aSzSca.x;


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

    std::cout << "x: GMin,Gax " << GMin << " " << GMax << "\n";
    
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
    std::cout << "y: GMin,Gax " << GMin << " " << GMax << "\n";
    
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
    std::cout << "xy: GMin,Gax " << GMin << " " << GMax << "\n";
    
    aResXY = (aImXY.in() - GMin) * (255.0 / (GMax-GMin));
 
    ELISE_COPY
    (
        aTiffXY.all_pts(),
        aResXY,
        aTiffXY.out()
    );

    return EXIT_SUCCESS;
}

int DoTile_main(int argc,char ** argv)
{
/*    std::string aDirTmp = "Tmp-TIL", aPrefixName = "_TIL_";
    std::string aTmp;
    std::string aIm1Name, aIm2Name;
    std::string aNameType;
    eTypeImporGenBundle aType;

    int aK1,aK2;
    bool aGraph=false;
 
    Pt2di aImTilSz(5000,3500), aImTilSzTmp(0,0), aImTilGrid(0,0);

    GenIm::type_el aTypeOut = GenIm::u_int1;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aIm1Name,"First Image RPC file")
	           << EAMC(aIm2Name,"Second Image RPC file"),
        LArgMain() << EAM(aNameType,"Type",true,"Type of sensor (see eTypeImporGenBundle)")
                   << EAM(aImTilSz,"Sz",true,"Tile size of the left image, Def Sz=[5000,3000]")	
	           << EAM(aGraph,"Graph",true,"Create a graphom xml for Tapioca, Def=false")
    );


    bool aModeHelp;
    StdReadEnum(aModeHelp,aType,aNameType,eTIGB_NbVals);

    //graphHom
    std::string aGHOut = "//GraphHomSat.xml";
    cSauvegardeNamedRel aGH;

    CameraRPC aCRPC1(aIm1Name,aType);
    CameraRPC aCRPC2(aIm2Name,aType);

    Tiff_Im aIm1 = Tiff_Im::StdConv(aCRPC1.GetImName() + std::string(".tif"));
    Tiff_Im aIm2 = Tiff_Im::StdConv(aCRPC2.GetImName() + std::string(".tif"));


    //aImTilGrid.x = round_up(double(aCRPC1.SzBasicCapt3D().x)/aImTilSz.x);
    //aImTilGrid.y = round_up(double(aCRPC1.SzBasicCapt3D().y)/aImTilSz.y);
    aImTilGrid.x = round_up(double(aIm1.sz().x)/aImTilSz.x);
    aImTilGrid.y = round_up(double(aIm1.sz().y)/aImTilSz.y);
  

    ELISE_fp::MkDirSvp(aDirTmp);

    //(0) divide the imgs to tiles
    for(aK1=0; aK1<aImTilGrid.x; aK1++)
    {
        for(aK2=0; aK2<aImTilGrid.y; aK2++)
	{

	    //check if marginal tiles (this shall include image validity)
	    if(aK1 < (aImTilGrid.x-1))
	        aImTilSzTmp.x = aImTilSz.x;
	    else
	        aImTilSzTmp.x = aIm1.sz().x - aK1*aImTilSz.x;//aCRPC1.SzBasicCapt3D().x - aK1*aImTilSz.x;

	    if(aK2 < (aImTilGrid.y-1))
	        aImTilSzTmp.y = aImTilSz.y;
	    else
	        aImTilSzTmp.y = aIm1.sz().y - aK2*aImTilSz.y; //aCRPC1.SzBasicCapt3D().y - aK2*aImTilSz.y;


            aTmp = aDirTmp + "//" + aIm1Name.substr(0,aIm1Name.size()-4) + 
		   aPrefixName + ToString(aK1) + "_" + ToString(aK2) + 
                   "_Pt_" + ToString(aK1*aImTilSzTmp.x) + "_" + 
                   ToString(aK2*aImTilSzTmp.y) + ".tif";

            Tiff_Im aTilCur = Tiff_Im
	    (
                aTmp.c_str(),
	        aImTilSzTmp,
	        aTypeOut,
	        Tiff_Im::No_Compr,
	        aIm1.phot_interp(),
	        Tiff_Im::Empty_ARG
	    );
		           
	    ELISE_COPY
	    (
                aTilCur.all_pts(),
	        trans(aIm1.in(), Pt2di(aK1*aImTilSzTmp.x, aK2*aImTilSzTmp.y)), 
	        aTilCur.out()
	    );
	}
   }

   // For eqch tile in the left image,
// (a) define the volume in 3D space that it sees, and
  //  * (b) project that volume to the second image;
    // (c) form an image to contain the backprojected zone
    // (d) save the the pair to an XML file   

    //(a) corner start in top left corner and 
    //    follow clockwise direction
    //    V3D1H - 1st corner in 3D at height=last_height
    //    V3D1L - 1st corner in 3D at height=first_height
    //    V2D1  - 1st corner in 2D
    Pt3dr aV3D1H, aV3D2H, aV3D3H, aV3D4H;
    Pt3dr aV3D1L, aV3D2L, aV3D3L, aV3D4L;
    Pt2dr aV2D1, aV2D2, aV2D3, aV2D4, aV2DTmp;
   

    for(aK1=0; aK1<aImTilGrid.x; aK1++)
    {
        for(aK2=0; aK2<aImTilGrid.y; aK2++)
	{
            Pt2dr aMax, aMin;

	    //check if marginal tiles (this shall include imge validity)
	    if(aK1 < (aImTilGrid.x-1))
	    {
		aV2D2.x = (aK1+1)*aImTilSz.x;
		aV2D3.x = (aK1+1)*aImTilSz.x;
	    }
	    else
	    {
	        aV2D2.x = aIm2.sz().x - 1;//aCRPC2.SzBasicCapt3D().x -1;
	        aV2D3.x = aIm2.sz().x - 1;//aCRPC2.SzBasicCapt3D().x -1;
	    }
	    if(aK2 < (aImTilGrid.y-1))
	    {
		aV2D3.y = (aK2+1)*aImTilSz.y;
		aV2D4.y = (aK2+1)*aImTilSz.y;
	    }
	    else
	    {
		aV2D3.y = aIm2.sz().y - 1;//aCRPC1.SzBasicCapt3D().y -1;
		aV2D4.y = aIm2.sz().y - 1;//aCRPC1.SzBasicCapt3D().y -1;
	    }

            aV2D1.x = aK1*aImTilSz.x;
	    aV2D4.x = aK1*aImTilSz.x;
	    aV2D1.y = aK2*aImTilSz.y;
	    aV2D2.y = aK2*aImTilSz.y;


	    //3d volume
	    aV3D1H = aCRPC1.ImEtZ2Terrain(aV2D1, aCRPC1.GetRPC()->last_height);
	    aV3D1L = aCRPC1.ImEtZ2Terrain(aV2D1, aCRPC1.GetRPC()->first_height);
            
	    aV3D2H = aCRPC1.ImEtZ2Terrain(aV2D2, aCRPC1.GetRPC()->last_height);
	    aV3D2L = aCRPC1.ImEtZ2Terrain(aV2D2, aCRPC1.GetRPC()->first_height);

	    aV3D3H = aCRPC1.ImEtZ2Terrain(aV2D3, aCRPC1.GetRPC()->last_height);
	    aV3D3L = aCRPC1.ImEtZ2Terrain(aV2D3, aCRPC1.GetRPC()->first_height);

	    aV3D4H = aCRPC1.ImEtZ2Terrain(aV2D4, aCRPC1.GetRPC()->last_height);
	    aV3D4L = aCRPC1.ImEtZ2Terrain(aV2D4, aCRPC1.GetRPC()->first_height);


	    //backproject to aCRPC2
            bool IS_INI=false;
	    if(aCRPC2.PIsVisibleInImage(aV3D1H))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D1H);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )//becase PIsVisibleInImage checks the entire img
		    CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    

	    }

	    if(aCRPC2.PIsVisibleInImage(aV3D1L))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D1L);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )
	            CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    	
	    }

	    if(aCRPC2.PIsVisibleInImage(aV3D2H))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D2H);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )
	            CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    	
	    }
	    if(aCRPC2.PIsVisibleInImage(aV3D2L))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D2L);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )
	            CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    	
	    }

	    if(aCRPC2.PIsVisibleInImage(aV3D3H))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D3H);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )
	            CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    	
	    }
	    if(aCRPC2.PIsVisibleInImage(aV3D3L))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D3L);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )
	            CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    	
	    }

	    if(aCRPC2.PIsVisibleInImage(aV3D4H))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D4H);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )
	            CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    	
	    }
	    if(aCRPC2.PIsVisibleInImage(aV3D4L))
	    {
                aV2DTmp = aCRPC2.Ter2Capteur(aV3D4L);
	        if( aIm2.sz().x > aV2DTmp.x && aIm2.sz().y > aV2DTmp.y )
	            CheckBounds(aMin, aMax, aV2DTmp, IS_INI);    	
	    }
	
	    if(IS_INI)
	    {
		aMin.x = round_down(aMin.x);
		aMax.x = round_down(aMax.x);
		aMin.y = round_down(aMin.y);
		aMax.y = round_down(aMax.y);
                
		aImTilSzTmp.x = (aMax.x - aMin.x);
                aImTilSzTmp.y = (aMax.y - aMin.y);
	        
		//save img
                aTmp = aDirTmp + "//" + aIm2Name.substr(0,aIm2Name.size()-4) + 
                       aPrefixName + ToString(aK1) + "_" + ToString(aK2) + 
                       "_Pt_" + ToString(int(aMin.x)) + "_" + ToString(int(aMin.y)) + 
                       ".tif";


		Tiff_Im aTilCur = Tiff_Im
	        (
                    aTmp.c_str(),
	            aImTilSzTmp,
	            aTypeOut,
	            Tiff_Im::No_Compr,
	            aIm1.phot_interp(),
	            Tiff_Im::Empty_ARG
	        );


	        ELISE_COPY
	        (
                    aTilCur.all_pts(),
	            trans(aIm2.in(), Pt2di(aMin.x,aMin.y)), 
	            aTilCur.out()
	        );

                //std::cout << "*" << aTmp << " " << "minx/y=" << aMin << std::endl; 

                //save GraphHom
		if(aGraph)
                   aGH.Cple().push_back(cCpleString( aIm1Name.substr(0,aIm1Name.size()-4) + 
					           aPrefixName + ToString(aK1) + "_" + 
						   ToString(aK2) + ".tif", 
						   aIm2Name.substr(0,aIm2Name.size()-4) + 
						   aPrefixName + ToString(aK1) + "_" + 
						   ToString(aK2) + ".tif"));

	    }
	}
    }
    if(aGraph)
        MakeFileXML(aGH,aDirTmp+aGHOut);

    //(1) run PASTIS with pairs of tiles
    //std::string aTapRun = "mm3d Tapioca File " + aDirTmp+aGHOut + " -1 ExpTxt=-1";
    //System(aTapRun,true);
*/
    return EXIT_SUCCESS;
}


void CheckBounds(Pt2dr & aPmin, Pt2dr & aPmax, const Pt2dr & aP, bool & IS_INI)
{

    if(IS_INI)
    {
        if(aP.x < aPmin.x) 
           aPmin.x = aP.x;
        if(aP.x > aPmax.x)
    	   aPmax.x = aP.x;
	if(aP.y < aPmin.y) 
	   aPmin.y = aP.y;
	if(aP.y > aPmax.y)
	   aPmax.y = aP.y;

    }
    else
    {
        aPmin = aP;
        aPmax = aP;

	IS_INI = true;
    }


}

//some tests taken from intro0
int TestER_main5(int argc,char ** argv)
{
    std::string aImName;

    ElInitArgMain
    (
        argc, argv,
	LArgMain() << EAMC(aImName,"Image name"),
	LArgMain()
    );

    Pt2di SZ(256,256);

    Tiff_Im aIm = Tiff_Im::StdConv(aImName);
    Im2D_U_INT1 I(256,256);

    Gray_Pal  Pgr (30);
    RGB_Pal   Prgb  (5,5,5);
    Disc_Pal  Pdisc = Disc_Pal::PNCOL();
    Circ_Pal  Pcirc = Circ_Pal::PCIRC6(30);

    Elise_Set_Of_Palette SOP(NewLElPal(Pdisc)+Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));

    Video_Display Ecr((char *) NULL);
    Ecr.load(SOP);
    Video_Win   W  (Ecr,SOP,Pt2di(50,50),Pt2di(SZ.x,SZ.y));

    W.set_title("Une fenetre");

    ELISE_COPY
    (
        I.all_pts(),
	aIm.in(),
        W.out(Prgb)
    );
    getchar();

    std::cout << "dddddddffff" << "\n";

    return EXIT_SUCCESS;
}

//test OpticalCenterOfPixel
int TestER_main100(int argc,char ** argv)
{
    std::string aFullName;
    std::string aDir;
    std::string aNameOri;
    std::list<std::string> aListFile;

    std::string aNameType;
    eTypeImporGenBundle aType;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName,"Orientation file full name (Dir+OriPattern)"),
	LArgMain() << EAM(aNameType,"Type",true,"Type of sensor (see eTypeImporGenBundle)",eSAM_None,ListOfVal(eTT_NbVals,"eTT_"))
    );

    std::cout << aFullName << std::endl;
   
    bool aModeHelp;
    StdReadEnum(aModeHelp,aType,aNameType,eTIGB_NbVals);

    return EXIT_SUCCESS;
}

//test homography
int TestER_hom_main(int argc,char ** argv)
{
    cElMap2D       * aMap = cElMap2D::FromFile("homography2.xml");
    //cElHomographie * aH(aMap);
    cElMap2D * aMapI = aMap->Map2DInverse();
    aMapI->ToXmlGen();
    MakeFileXML(aMapI->ToXmlGen(),"homography2inv.xml");
    //aH->Show();

    return EXIT_SUCCESS;
}


//test export of a CamStenope into bundles of rays
int TestER_main2(int argc,char ** argv)
{

	TestEllips_3D();
		/*    cInterfChantierNameManipulateur * aICNM;
    std::string aFullName;
    std::string aDir;
    std::string aNameOri;
    std::list<std::string> aListFile;

    Pt2di aGridSz;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName,"Orientation file full name (Dir+OriPattern)"),
        LArgMain() << EAM(aGridSz,"GrSz",true)
    );
    
    SplitDirAndFile(aDir, aNameOri, aFullName);

    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aNameOri);


    for(std::list<std::string>::iterator itL = aListFile.begin(); itL != aListFile.end(); itL++ )
    {
        CamStenope * aCurCamSten = CamStenope::StdCamFromFile(true, aDir+(*itL), aICNM);
        aCurCamSten->ExpImp2Bundle(aGridSz, *itL);
    }
*/
    return EXIT_SUCCESS;
}

int TestER_rpc_main(int argc,char ** argv)
{
    std::string aFullName1,aFullName2;
    Pt2dr aP1, aP2, aP1_, aP2_;

    std::vector<double> aVPds;
    std::vector<ElSeg3D> aVS;
    
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName1,"Orientation file1")
                   << EAMC(aFullName2,"Orientation file2")
                   << EAMC(aP1,"P1")
                   << EAMC(aP2,"P2"),
        LArgMain()
    );
    std::cout << "in: " << aP1 << ", " << aP2 << "\n";

    CameraRPC aRPC1(aFullName1);
    CameraRPC aRPC2(aFullName2);

    double aZ1 = 100;//aRPC1.GetAltiSol() + double(aRPC1->GetGrC32() - aRPC1->GetGrC31())/2;

    Pt3dr aPt1El1 = aRPC1.ImEtZ2Terrain(aP1,aZ1);
    Pt3dr aPt2El1 = aRPC1.ImEtZ2Terrain(aP1,aZ1+10);
    Pt3dr aPt1El2 = aRPC2.ImEtZ2Terrain(aP2,aZ1);
    Pt3dr aPt2El2 = aRPC2.ImEtZ2Terrain(aP2,aZ1+10);

    ElSeg3D aElS1(aPt1El1,aPt2El1); //aRPC1.Capteur2RayTer(aP1);
    ElSeg3D aElS2(aPt1El2,aPt2El2); //aRPC2.Capteur2RayTer(aP2);
    std::cout << "el: " << aElS1.P0() << " " << aRPC1.ImEtZ2Terrain(aP1,aZ1) << ", " << aElS1.P1() << "\n";
    std::cout << "el: " << aElS2.P0() << " " << aRPC2.ImEtZ2Terrain(aP2,aZ1) << ", " << aElS2.P1() << "\n";
    
    aVS.push_back(aElS1);
    aVS.push_back(aElS2);
    aVPds.push_back(1.0);
    aVPds.push_back(1.0);

    bool aIsOK;
    Pt3dr aRes = ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK);

    aP1_ = aRPC1.Ter2Capteur(aRes);        
    aP2_ = aRPC2.Ter2Capteur(aRes);        

    std::cout << "in: " << aP1 << ", pred: " << aP1_ << "\n"
                 "    " << aP2 << ", pred: " << aP2_ << "\n"
                 "R3= " << aRes << "\n" ;

    return EXIT_SUCCESS;
}

int TestER_grille_main(int argc,char ** argv)
{
    std::string aFullName, aChSysStr;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName,"OrientationGrille file"),
        LArgMain() << EAM(aChSysStr,"ChSys",true)
    );

    /*bool aModeHelp;
    eTypeImporGenBundle aType;
    std::string aNameType="TIGB_MMOriGrille";
    StdReadEnum(aModeHelp,aType,aNameType,eTIGB_NbVals);

    cSystemeCoord * aChSys = new cSystemeCoord(StdGetObjFromFile<cSystemeCoord>
            (
                aChSysStr,
                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                "SystemeCoord",
                "SystemeCoord"
             ));
    */

    //CameraRPC aCamRPC(aFullName);
    bool aModeHelp;
    eTypeImporGenBundle aType;
    std::string aNameType="TIGB_MMEuclid";
    StdReadEnum(aModeHelp,aType,aNameType,eTIGB_NbVals);


    cSystemeCoord * aChSys = new cSystemeCoord(StdGetObjFromFile<cSystemeCoord>
            (
                aChSysStr,
                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                "SystemeCoord",
                "SystemeCoord"
            ));

    cRPC aRPC(aFullName,aType,aChSys); 
    
    //OrientationGrille aGrill(aFullName);
    //aGrill.GetRangeZ();

    return EXIT_SUCCESS;
}

class cTmpPileER
{
    public :
        cTmpPileER(int aK,float aZ,float aPds);
        // Caclul entre deux cellule successive le poids exponentiel 
       // qui sera utilise pour le filtrage recursif
        void SetPPrec(cTmpPileER &,float aExpFact);
        // double Pds() {return mPds0 / mNb0;}
        double ZMoy() {return mPds0 ? (mZP0/mPds0) : 0 ;}

    // private :
         int    mK;
         double mCpteur;
         double mZInit;
         double mPInit;
         double mZP0;   // Z Pondere par le poids
         double mPds0;  // Poids 
         double mNb0;   // si le PdsInit=1, alors mNb0==mPds0, compte le nombre de pt de chaque cluste
                        // (a une constante globale pres)

        // Variale pour le filtrage recursif "plus"
         double mNewZPp;
         double mNewPdsp;
         double mNewNbp;

        // Variale pour le filtrage recursif "moins"
         double mNewZPm;
         double mNewPdsm;
         double mNewNbm;

         double mPPrec;
         double mPNext;
         bool   mSelected;
};

cTmpPileER::cTmpPileER(int aK,float aZ,float aPds) :
	mK     (aK),
	mCpteur(0),
	mZInit (aZ),
	mPInit (aPds),
	mZP0   (aZ * aPds),
	mPds0  (aPds),
	mNb0   (1.0),
	mPPrec (-1),
	mPNext (-1),
	mSelected (false)
{}

void cTmpPileER::SetPPrec(cTmpPileER & aPrec,float aExpFact)
{
	ELISE_ASSERT(mZInit>=aPrec.mZInit,"Ordre coherence in cTmpPile::SetPPrec");

	double aPds = (aPrec.mZInit-mZInit)/aExpFact;
	aPds = exp(aPds);
    mPPrec  = aPds;
    aPrec.mPNext = aPds;

		
}

void FiltrageAllerEtRetourER(std::vector<cTmpPileER> & aVTmp)
{
	int aNb = (int)aVTmp.size();

	
	aVTmp[0].mNewPdsp = aVTmp[0].mPds0;
    aVTmp[0].mNewZPp  = aVTmp[0].mZP0;
    aVTmp[0].mNewNbp  = aVTmp[0].mNb0;

	//propagation de gauche vers droit
	std::cout << "propagation de gauche vers droit" << "\n";
	for (int aK=1 ; aK<int(aVTmp.size()) ; aK++)
	{
		aVTmp[aK].mNewPdsp = aVTmp[aK].mPds0 + aVTmp[aK-1].mNewPdsp * aVTmp[aK].mPPrec;
        aVTmp[aK].mNewZPp  = aVTmp[aK].mZP0  + aVTmp[aK-1].mNewZPp  * aVTmp[aK].mPPrec;
        aVTmp[aK].mNewNbp  = aVTmp[aK].mNb0  + aVTmp[aK-1].mNewNbp  * aVTmp[aK].mPPrec;

		std::cout << "P=" << aVTmp[aK].mPds0 << ", " << aVTmp[aK].mNewPdsp << ", " << aVTmp[aK].mPPrec << "\n" 
                  << "ZP="<< aVTmp[aK].mZInit << " " << aVTmp[aK].mZP0 << ", " << aVTmp[aK].mNewZPp << " dZ=" << aVTmp[aK-1].mNewZPp  * aVTmp[aK].mPPrec << "\n"
                  << "N=" << aVTmp[aK].mNb0 << " " << aVTmp[aK].mNewNbp << "\n\n";
	}

	//propagation de droit vers gauche
	std::cout << "propagation de droit vers gauche" << "\n";
	
	aVTmp[aNb-1].mNewPdsm = aVTmp[aNb-1].mPds0;
    aVTmp[aNb-1].mNewZPm = aVTmp[aNb-1].mZP0;
    aVTmp[aNb-1].mNewNbm = aVTmp[aNb-1].mNb0;

	for (int aK=(int)(aVTmp.size() - 2); aK>=0 ; aK--)
    {
    	aVTmp[aK].mNewPdsm = aVTmp[aK].mPds0 + aVTmp[aK+1].mNewPdsm * aVTmp[aK].mPNext;
        aVTmp[aK].mNewZPm  = aVTmp[aK].mZP0  + aVTmp[aK+1].mNewZPm  * aVTmp[aK].mPNext;
        aVTmp[aK].mNewNbm  = aVTmp[aK].mNb0  + aVTmp[aK+1].mNewNbm  * aVTmp[aK].mPNext;
		
		std::cout << "P=" << aVTmp[aK].mPds0 << ", " << aVTmp[aK].mNewPdsm << ", " << aVTmp[aK].mPNext << "\n" 
                  << "ZP="<< aVTmp[aK].mZInit << " " << aVTmp[aK].mZP0 << ", " << aVTmp[aK].mNewZPm << " dZ=" << aVTmp[aK+1].mNewZPm  * aVTmp[aK].mPNext << "\n"
                  << "N=" << aVTmp[aK].mNb0 << " " << aVTmp[aK].mNewNbm << "\n\n";
    }

     // Memorisation dans mZP0 etc.. du resultat (droite + gauche - VCentrale) , VCentrale a ete compte deux fois
     for (int aK=0 ; aK<int(aVTmp.size()) ; aK++)
     {
          aVTmp[aK].mZP0  = (aVTmp[aK].mNewZPp  + aVTmp[aK].mNewZPm  - aVTmp[aK].mZP0) / aVTmp.size();
          aVTmp[aK].mPds0 = (aVTmp[aK].mNewPdsp + aVTmp[aK].mNewPdsm - aVTmp[aK].mPds0) / aVTmp.size();
          aVTmp[aK].mNb0  = (aVTmp[aK].mNewNbp  + aVTmp[aK].mNewNbm  - aVTmp[aK].mNb0) / aVTmp.size();
		
		  std::cout << "P=" << aVTmp[aK].mNewPdsp << ", " << aVTmp[aK].mNewPdsm << ",->" << aVTmp[aK].mPds0  << "\n" 
                  << "ZP="<< aVTmp[aK].mZInit << " " << aVTmp[aK].mNewZPp  << " " << aVTmp[aK].mNewZPm << ", ->" << aVTmp[aK].mZP0/aVTmp[aK].mPds0 << "\n" //see ZMoy()
                  << "N=" << aVTmp[aK].mNewNbp << " " << aVTmp[aK].mNewNbm << " " << aVTmp[aK].mNb0 << "\n\n";
     }


}

int TestER_filtRec_main(int argc,char ** argv)
{

    std::string          aCelFich;
	std::vector<double>  aCel;
	double  			 aSig=5.0;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aCelFich,"FIle with the cellule vector"),
        LArgMain() << EAM(aSig,"Sig",true, "Gaussian sigma")
	);

	ELISE_fp aFIn(aCelFich.c_str(),ELISE_fp::READ);
	char * aLine = aFIn.std_fgets();

	char * it = NULL;
	for (it = aLine; *it != '\0'; ) 
	{
		double aTmp = std::atof(it);
    	aCel.push_back( aTmp );

		if(aTmp < 10)
			it=it+2;
		else if(aTmp < 100)
			it=it+3;
	
		std::cout << aTmp << "\n";
    }


	//construct a cellule
	std::vector<cTmpPileER> aVPil;
	float aZFact = (aSig*0.5) * sqrt(2.0);
	
	for(int aK=0; aK < (int)(aCel.size()); aK++)
	{
		aVPil.push_back(cTmpPileER(aK,aCel.at(aK),1.0));
		
		//set precision?
		if(aK>0)
		{
			std::cout << aVPil[aK].mZInit << " " << aVPil[aK-1].mZInit << "\n"; 
			aVPil[aK].SetPPrec(aVPil[aK-1],aZFact);


		}
	}

	//recursive filtering
	FiltrageAllerEtRetourER(aVPil);
	FiltrageAllerEtRetourER(aVPil);



    return EXIT_SUCCESS;
}



class anAppli_PFM2Tiff
{
    public:
        anAppli_PFM2Tiff(int argc,char ** argv);

        int DoExe();


    private: 
        int DoPFM();
        int DoTif();

        int ReadPFMHeader(FILE *Data);
        void SkipSpace(FILE *Data);
        void SkipComment(FILE *Data);

        std::string mInName;
        std::string mOutName;
        Pt2di       mSz;
};


//Check whether machine is little endian
int IsLittleEndian()
{
    int intval = 1;
    unsigned char *uval = (unsigned char *)&intval;
    return uval[0] == 1;
}

anAppli_PFM2Tiff::anAppli_PFM2Tiff(int argc,char ** argv) : 
       mInName(""),
       mOutName(""),
       mSz(Pt2di(0,0))
{

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mInName,"Input file"),
        LArgMain() << EAM(mOutName,"Out",true, "Output pfm or tif file")
	);

    /* Determine what conversion to make */
    const char * aExt = strrchr(mInName.c_str(), '.');

    if (strcmp(aExt, ".PFM") == 0 || strcmp(aExt, ".pfm") == 0)
    {
        DoTif();

    }
    else if (strcmp(aExt, ".TIF") == 0 || strcmp(aExt, ".tif") == 0)
    {
        DoPFM();

    }
    else 
    {
        ELISE_ASSERT(false,"PFM2Tiff_main image format not supported?");
    }
}

int anAppli_PFM2Tiff::DoExe()
{

    return EXIT_SUCCESS;
}

void anAppli_PFM2Tiff::SkipSpace(FILE *Data)
{
    char aC;

    do
    { 
        aC = std::getc(Data);
    }
    while (aC == '\n' || aC == ' ' || aC == '\t' || aC == '\r');
    
    std::ungetc(aC, Data);    

}

void anAppli_PFM2Tiff::SkipComment(FILE *Data)
{
    char aC;

    while ((aC=std::getc(Data)) == '#')
        while (std::getc(Data) != '\n') ;
    std::ungetc(aC, Data);

}

int anAppli_PFM2Tiff::ReadPFMHeader(FILE *Data)
{
    char aC;

    if (std::getc(Data) != 'P' || std::getc(Data) != 'f')
        ELISE_ASSERT(false,"anAppli_PFM2Tiff::ReadPFMHeader(): wrong header code");

    SkipSpace(Data);
    SkipComment(Data);

    SkipSpace(Data);
    if( std::fscanf(Data,"%d",&mSz.x) == 0 )
        ELISE_ASSERT(false,"anAppli_PFM2Tiff::ReadPFMHeader(): can't read img size");
    SkipSpace(Data);
    if (std::fscanf(Data,"%d",&mSz.y) == 0)
        ELISE_ASSERT(false,"anAppli_PFM2Tiff::ReadPFMHeader(): can't read img size");
    
    aC = getc(Data);
    if (aC == '\r') 
        aC = std::getc(Data);

    if (aC != '\n') 
    {
        if (aC == ' ' || aC == '\t' || aC == '\r')
        {
            ELISE_ASSERT(false,"anAppli_PFM2Tiff::ReadPFMHeader(): newline expected in file after image height");
        }
        else
            ELISE_ASSERT(false,"anAppli_PFM2Tiff::ReadPFMHeader(): whitespace expected in file after image height");
    }


    return EXIT_SUCCESS;
}

int anAppli_PFM2Tiff::DoTif()
{
    if (! EAMIsInit(&mOutName))
        mOutName = StdPrefix(mInName) + ".tif";

    FILE *aFO = fopen(mInName.c_str(), "rb"); 
    if (aFO == 0)
        ELISE_ASSERT(false,"anAppli_PFM2Tiff::DoTif(): could not open file");

    ReadPFMHeader(aFO);
    

    SkipSpace(aFO);
    
    float aSc;
    if (fscanf(aFO,"%f",&aSc)) //scale factor (if negative, little endian)
        std::cout << "no scale read" << "\n";        
//ELISE_ASSERT(false,"anAppli_PFM2Tiff::ReadPFMHeader(): can't read scale");

    char aC = getc(aFO);
    if (aC == '\r')
        aC = std::getc(aFO);

    //skip newline character
    if (aC != '\n')
    {
        if (aC == ' ' || aC == '\t' || aC == '\r')
        {
            ELISE_ASSERT(false,"anAppli_PFM2Tiff::DoTif(): newline expected in file after image height");
        }
        else
            ELISE_ASSERT(false,"anAppli_PFM2Tiff::DoTif(): whitespace expected in file after image height");
    }

    Im2D_REAL4 aRes(mSz.x,mSz.y);


/*    int LitEndF = (aSc < 0);
    int LitEndM = IsLittleEndian();
    int needSwap = (LitEndF!= LitEndM);
*/
    for (int aK2=mSz.y-1; aK2>=0; aK2--)
    {
        float * aLine = new float[mSz.x];
        if ((int)fread(aLine, sizeof(float), mSz.x, aFO) != mSz.x)
            ELISE_ASSERT(false,"anAppli_PFM2Tiff::DoTif(): File is too short ");

        for (int aK1=0; aK1<mSz.x; aK1++)
        {
            //std::cout << aLine[aK1] << " " << "\n";
            aRes.SetR(Pt2di(aK1,aK2),aLine[aK1]);
        }

        delete[] aLine;

    }
    fclose(aFO);

    ELISE_COPY
    (
        aRes.all_pts(),
        aRes.in(),
        Tiff_Im(
            mOutName.c_str(),
            aRes.sz(),
            GenIm::real4,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero,
            Tiff_Im::Empty_ARG ).out()
    );

 

    return EXIT_SUCCESS;
}

int anAppli_PFM2Tiff::DoPFM()
{
    if (! EAMIsInit(&mOutName))
        mOutName = StdPrefix(mInName) + ".pfm";

    Tiff_Im ImgTiff = Tiff_Im::StdConvGen(mInName,-1,true);

    Im2D_REAL4 I(ImgTiff.sz().x, ImgTiff.sz().y);
    ELISE_COPY
    (
        I.all_pts(),
        ImgTiff.in(),
        I.out()
    );
   
    mSz = ImgTiff.sz();
    
    //save
    FILE *stream = fopen(mOutName.c_str(), "wb");
    fprintf(stream, "Pf\n%d %d\n%f\n", mSz.x, mSz.y, float(-1.0));

    for (int aK2=mSz.y-1; aK2>=0; aK2--)
    {
        float *aLine = new float[mSz.x]; 
        for (int aK1=0; aK1<mSz.x; aK1++)
        {
            aLine[aK1] = float(I.Val(aK1,aK2));
        }
        //std::cout << aLine[0] << " " << aLine[10] << "\n";
 
        if(int(fwrite(aLine, sizeof(float), mSz.x, stream)) != mSz.x)
            ELISE_ASSERT(false,"File is too short");

        delete[] aLine;
    }
    fclose(stream); 

    return EXIT_SUCCESS;

}


int PFM2Tiff_main(int argc,char ** argv)
{
    anAppli_PFM2Tiff anAppli(argc,argv);
    anAppli.DoExe();
    
    
    return EXIT_SUCCESS;

}

class cBAL2OriMicMac
{
    public :
        cBAL2OriMicMac(int argc,char ** argv);
        ~cBAL2OriMicMac() { delete mMulPts; };

        bool DoAll();


    private :
        std::string mName;
        std::string mOri;
        std::string mDir;
        
        cInterfChantierNameManipulateur * mICNM;    
        std::string                       mKeyOri2Im;

        cSetTiePMul                       * mMulPts;
        Pt2dr                               mPP; //will be calculated as perfectly in the middle
        Pt2di                               mSz;
                                         
        int                                 mNumCam;
        int                                 mNumObs;
        int                                 mNumPts;
                                         
        double                              mAltiSol;//based on 3D pts
        double                              mProf;

        std::map<int,double*>               mPoses;   
        std::map<int,std::vector<Pt2dr>* >  mPtIdTr;  
        std::map<int,std::vector<int>* >    mPtIdPId; 

        std::vector<Pt3dr>                  mPt3D;
        std::vector<std::string>            mCamNames;        

        void SetPP(Pt2dr& PP) { mPP=PP; };
        void SetSz(Pt2di& Sz) { mSz=Sz; };
        void SetAltiSol(const double& A) { mAltiSol=A; };
        void SetProf(const double& P) { mProf=P; };

        bool ReadBAL();
        void SaveMicMac();
        bool ReadMicMac();
        bool SaveBAL();

        template <typename T>
        void FileReadOK(FILE *fptr, const char *format, T *value);

        std::string GenCamName(const int&);
        void ShiftHomol(const Pt2dr&,bool toPP);
};


void cBAL2OriMicMac::ShiftHomol(const Pt2dr& aPP,bool toPP)
{
    for (auto aPIdx : mPtIdTr)
    {
        for (auto & aTrack : (*aPIdx.second))
        {
            if (!toPP) //to Micmac image coordinate system
            {
                aTrack.x += aPP.x;
                aTrack.y += aPP.y;
            }
            else //PP at (0,0)
            {
                aTrack.x -= aPP.x;
                aTrack.y -= aPP.y;
                
            }
        }
    }
}


std::string cBAL2OriMicMac::GenCamName(const int& aName)
{
    return("Img-" + ToString(aName) + ".tif");
}

template <typename T>
void cBAL2OriMicMac::FileReadOK(FILE *fptr, const char *format, T *value)
{
    int OK = fscanf(fptr, format, value);
    if (OK != 1)
        ELISE_ASSERT(false, "cBAL2OriMicMac::Read")
}

void cBAL2OriMicMac::SaveMicMac()
{

    /* Save empty images */
    for (int aCam=0; aCam<mNumCam; aCam++)
    {
        ELISE_fp aFPOut(mCamNames.at(aCam).c_str(),ELISE_fp::WRITE);
        aFPOut.close();
    }
    

    /* Homol */
    mMulPts = new cSetTiePMul(0,&mCamNames);
    for (int aK=0; aK<int(mPtIdTr.size()); aK++)
    {
        std::vector<float> aAttr;
        mMulPts->AddPts(*mPtIdPId[aK],*mPtIdTr[aK],aAttr);
    }
    
    ELISE_fp::MkDir("Homol");
    mMulPts->Save("Homol/PMul.txt");
    

    /* Poses */
    for (int aCam=0; aCam<mNumCam; aCam++)
    {
        double * aCal = mPoses[aCam];
     
        Pt3dr aC  (aCal[3],aCal[4],aCal[5]);
        Pt3dr aAng(aCal[0],aCal[1],aCal[2]);
        ElRotation3D aOri(aC,aCal[0],aCal[1],aCal[2]);


        cOrientationConique aOriCon;
        cOrientationExterneRigide aOriRig;

        /* Le centre */
        aOriRig.Centre()        = aC;
        

        /* Les rotations */
        cRotationVect aRot;
        aRot.CodageAngulaire()  = aAng;

        aOriRig.ParamRotation() = aRot;
        aOriRig.AltiSol()       = mAltiSol; 
        aOriRig.Profondeur()    = mProf;
        aOriRig.KnownConv()     = eConvApero_DistM2C;//eConvAngPhotoMDegre;//eConvApero_DistM2C;
        aOriRig.IncCentre()     = Pt3dr(1,1,1);   

        aOriCon.TypeProj()      = eProjStenope;
        aOriCon.Externe()       = aOriRig;
        //aOriCon.Externe()       = From_Std_RAff_C2M(aOri,false); //aOriRig;

        cConvOri          aKOri;
        cConvExplicite    aKOriEx;

        
        aKOriEx.MatrSenC2M() = false;
        aKOriEx.Convention()   = eConvApero_DistM2C;
        aKOri.ConvExplicite()  = aKOriEx;
        
        //aKOri.KnownConv()       = eConvApero_DistC2M;//eConvAngPhotoMDegre;//eConvApero_DistM2C;
        aOriCon.ConvOri()       = aKOri;

        /* Autres */
        

        /* Camera calibration */
        cCalibrationInternConique aInCal;
        aInCal.KnownConv() = eConvApero_DistM2C;
        aInCal.F()    = aCal[6];
        aInCal.PP()   = mPP;
        aInCal.SzIm() = mSz;
        std::vector<double> aR1R3;
        aR1R3.push_back(aCal[7]);
        aR1R3.push_back(aCal[8]);

        std::vector< cCalibDistortion > aDistVec;
        cCalibDistortion                aDist;
       
        cCalibrationInterneRadiale aDistRad; 
/*        cCalibrationInternePghrStd aDistRad;
        aDistRad.P1() = 
        aDistRad.P2() = */
        aDistRad.CoeffDist() = aR1R3; 
        aDistRad.CDist()= mPP;

        aDist.ModRad() = aDistRad;
        
        aDistVec.push_back(aDist);
        aInCal.CalibDistortion() = aDistVec;

        std::string aCalOri  = mICNM->Assoc1To1(mKeyOri2Im,mCamNames.at(aCam),true);
        std::string aCalName = DirOfFile(aCalOri) + "AutoCal_Foc-50000.xml";
        MakeFileXML(aInCal,aCalName);

        aOriCon.FileInterne() = aCalName;
        MakeFileXML(aOriCon,aCalOri);

        delete[] aCal;
    }

}

bool cBAL2OriMicMac::SaveBAL()
{
    FILE* fptr = fopen("BAL-pb.txt", "w"); 
    if (fptr == NULL) {
      return false;
    };

    fprintf(fptr, "%d %d %d\n", mNumCam, mNumPts, mNumObs);

    /* Homol */
    ShiftHomol(mPP,true);

    int aNum;
    for (auto aT : mPtIdTr)
    {
        aNum=0;
        for (auto aP : *aT.second)
        {
            fprintf(fptr, "%d %d %g %g\n", mPtIdPId[aT.first]->at(aNum), aT.first, aP.x, aP.y);
            aNum++;
        }
    }
    
    /* Poses */
    for (auto aP : mPoses)
    {
        fprintf(fptr, "%g\n", aP.second[0]); 
        fprintf(fptr, "%g\n", -aP.second[1]); //this is to be clarified, for some reason the From_Std_RAff_C2M changes the sign :?
        fprintf(fptr, "%g\n", aP.second[2]); 
        fprintf(fptr, "%g\n", aP.second[3]); 
        fprintf(fptr, "%g\n", aP.second[4]); 
        fprintf(fptr, "%g\n", aP.second[5]); 
        fprintf(fptr, "%g\n", aP.second[6]); 
        fprintf(fptr, "%g\n", aP.second[7]); 
        fprintf(fptr, "%g\n", aP.second[8]); 
    } 

    /* 3D points */
    for (auto aPt : mPt3D) 
    {
        fprintf(fptr, "%g\n", aPt.x);
        fprintf(fptr, "%g\n", aPt.y);
        fprintf(fptr, "%g\n", aPt.z);
    }

    fclose(fptr);

    return true;
    
}

// Part of the read code that reads BAL problem is taken from ceres exemples 
bool cBAL2OriMicMac::ReadBAL()
{
    FILE* fptr = fopen(mName.c_str(), "r"); 
    if (fptr == NULL) {
      return false;
    };

    //int aNumParam;


    FileReadOK(fptr, "%d", &mNumCam);
    FileReadOK(fptr, "%d", &mNumPts);
    FileReadOK(fptr, "%d", &mNumObs);

    /* Initialize tracks */
    for (int aK=0; aK<mNumPts; aK++)
    {
        mPtIdTr[aK] = new std::vector<Pt2dr>();
        mPtIdPId[aK] = new std::vector<int>();
    }

    //aNumParam = 9*aNumCam + 3*aNumPts;


    /* Observations 2D */
    int aCamIdx,aPtIdx;
    Pt2dr aMinPt(1e10,1e10);
    Pt2dr aMaxPt(-1e10,-1e10);
    for (int aK=0; aK<mNumObs; aK++)
    {
        FileReadOK(fptr, "%d", &aCamIdx);
        FileReadOK(fptr, "%d", &aPtIdx);

    
        double aPt[2];
        for (int j = 0; j < 2; ++j) 
        {
            FileReadOK(fptr, "%lf", aPt + j);
        }
        

        std::vector<Pt2dr>* aTr = mPtIdTr[aPtIdx];
        aTr->push_back(Pt2dr(aPt[0],aPt[1]));


        std::vector<int>* aTrId = mPtIdPId[aPtIdx];
        aTrId->push_back(aCamIdx);

       
        aMinPt.x = ElMin(aMinPt.x,aPt[0]); 
        aMinPt.y = ElMin(aMinPt.y,aPt[1]); 

        aMaxPt.x = ElMax(aMaxPt.x,aPt[0]); 
        aMaxPt.y = ElMax(aMaxPt.y,aPt[1]); 

    }
    Pt2di aSz (round_ni(aMaxPt.x)-round_ni(aMinPt.x),
               round_ni(aMaxPt.y)-round_ni(aMinPt.y));
    SetSz(aSz);
    Pt2dr aPP(0.5*aSz.x,0.5*aSz.y);
    SetPP(aPP);

    //observations are shifted to bring them to positive values
    ShiftHomol(mPP,false);
        

    /* Camera parameters */
    for (int aCam=0; aCam<mNumCam; aCam++)
    {
        mPoses[aCam] = new double[9];
        double * aCal = mPoses[aCam];
        for (int aParam=0; aParam<9; aParam++)
        {
            FileReadOK(fptr, "%lf", aCal + aParam);
        }
    
        /* Create image names */
        mCamNames.push_back(GenCamName(aCam));     
    }


    

    /* 3D points en s'en fout pour l'instant ? */
    Pt3dr  aSom;
    double aVal;
    for (int aP=0; aP<mNumPts; aP++)
    {
        //X
        FileReadOK(fptr, "%lf", &aVal);
        aSom.x+=aVal;
        //Y
        FileReadOK(fptr, "%lf", &aVal);
        aSom.y+=aVal;
        //Z
        FileReadOK(fptr, "%lf", &aVal);
        aSom.z+=aVal;
    }
    aSom.x /= mNumPts;
    aSom.y /= mNumPts;
    aSom.z /= mNumPts;
    SetAltiSol(aSom.z);
    double * aPose0 = mPoses[0];
    SetProf(abs(aSom.z - aPose0[5]));

    std::cout << "Centroid = " << aSom << " AltSol=" << mAltiSol << ", Prof=" << mProf << "\n";
   
    fclose(fptr);
 
    return true;

}

bool cBAL2OriMicMac::ReadMicMac()
{

    std::list<std::string> aListFile;
    SplitDirAndFile(mDir, mOri, mName);
    aListFile = mICNM->StdGetListOfFile(mOri);



    for (auto itL : aListFile)
    {   
        mCamNames.push_back(itL);
    }
    mNumCam = int(mCamNames.size());


    const std::vector<std::string> *  aPMulVec = cSetTiePMul::StdSetName(mICNM,"",false);
    mMulPts = cSetTiePMul::FromFiles(*aPMulVec,&mCamNames);

    /* Poses */
    std::map<int,CamStenope*> aCams;
    for (auto itL : aListFile)
    {

        cCelImTPM * aMulPtsIm = mMulPts->CelFromName(itL);

        if(aMulPtsIm)
        {
            std::string aInOri     = mICNM->Assoc1To1(mKeyOri2Im,itL,true);
            aCams[aMulPtsIm->Id()] = CamOrientGenFromFile(aInOri,mICNM);
        }

    }
    
    /* Initialize poses structure */
    for (int aP=0; aP<mNumCam; aP++)
    {
        mPoses[aP] = new double[9]; //aPose;     

        //ElRotation3D aAngCP = aCams[aP]-> Orient();

        mPoses[aP][0] = aCams[aP]->Orient().teta01() * (180.0/PI);
        mPoses[aP][1] = aCams[aP]->Orient().teta02() * (180.0/PI); 
        mPoses[aP][2] = aCams[aP]->Orient().teta12() * (180.0/PI); 
        mPoses[aP][3] = aCams[aP]->Orient().tr().x; 
        mPoses[aP][4] = aCams[aP]->Orient().tr().y; 
        mPoses[aP][5] = aCams[aP]->Orient().tr().z; 
        mPoses[aP][6] = aCams[aP]->Focale();


        //also do image dependent camera calib


        cCalibrationInternConique      InCal = aCams[aP]->ExportCalibInterne2XmlStruct(aCams[aP]->SzBasicCapt3D()) ;
        std::vector<cCalibDistortion>  InCalDis = InCal.CalibDistortion();
        cCalibrationInterneRadiale     InCalRad = InCalDis.at(0).ModRad().Val();

        mPoses[aP][7] = InCalRad.CoeffDist().at(0); 
        mPoses[aP][8] = InCalRad.CoeffDist().at(1);            
    


    }
    
    /* Get number of points */
    mNumPts=0;
    for (auto aConf : mMulPts->VPMul())
        mNumPts+=aConf->NbPts();

    /* Initialize tracks */
    for (int aK=0; aK<mNumPts; aK++)
    {
        mPtIdTr[aK] = new std::vector<Pt2dr>();
        mPtIdPId[aK] = new std::vector<int>();
    }
    

    /* Homol : fill in the track structures */
    int aPos=0;
    for (auto aConf : mMulPts->VPMul())
    {
        aConf->IntersectBundle(aCams,mPtIdTr,mPtIdPId,mPt3D,aPos);

        aPos+=aConf->NbPts();
    }

    /* Get number of obs */ 
    mNumObs=0;
    for (auto aObs : mPtIdTr)
    {
        mNumObs += int(aObs.second->size());
    }

    return true;
}


bool cBAL2OriMicMac::DoAll()
{
    if (StdPostfix(mName)=="txt")
    {
        if (ReadBAL())
            SaveMicMac();
        else
            return false;
    }
    else
    {
        if (ReadMicMac())
            SaveBAL();
        else
            return false; 
    }

    return true;
}

cBAL2OriMicMac::cBAL2OriMicMac(int argc,char ** argv) :
    mDir("./"),
    mICNM(cInterfChantierNameManipulateur::BasicAlloc(mDir)),
    mPP(Pt2dr(0,0))
{
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mName,"File with BAL Problem or pattern of images")
                   << EAMC(mOri,"Input/Output orientation directory"),
        LArgMain() 
                   
    ); 
    //StdCorrecNameOrient(mOri,mDir);

    mKeyOri2Im = std::string("NKS-Assoc-Im2Orient@-") + mOri;
    
}

int BAL2OriMicMac_main(int argc,char ** argv)
{
    cBAL2OriMicMac aBal2MM(argc,argv);
    aBal2MM.DoAll();

    return 1.0;
}

//to do - ajouter la masq?

class cImDir
{
    public :
        cImDir(const std::string &aName);

        ElSeg3D & OC();
        const ElSeg3D & OC()const;

        Pt2dr & PP();
        const Pt2dr & PP()const;

        std::vector<ElSeg3D>        mDirs;
 
    private :
        const std::string           mName;
        ElSeg3D                     mOC;
        Pt2dr                       mPP;
};

cImDir::cImDir(const std::string &aName) :
     mName(aName)  ,
     mOC(ElSeg3D(Pt3dr(0,0,0),Pt3dr(0,0,0)))
{}
    
ElSeg3D & cImDir::OC()
{ return mOC; }


const ElSeg3D & cImDir::OC()const
{ return mOC; }

Pt2dr & cImDir::PP()
{ return mPP; }

const Pt2dr & cImDir::PP()const
{ return mPP; }

class Appli_ImPts2Dir
{
    public : 
        Appli_ImPts2Dir(int argc,char ** argv);
        
        int DoCalc();

    private :
        std::string                                   mDir;
        std::string                                   mOri;
        std::string                                   mIms;
        std::string                                   mOut;
        cInterfChantierNameManipulateur             * mICNM;
        const cInterfChantierNameManipulateur::tSet * mSetIm;

        int                                           mNbIm;
        int                                           mNbPts;

        std::vector<Pt2dr>                            mListPt2d;
        std::map<std::string,cImDir*>                 mMapImDirs;
       

        int Save(); 
};

Appli_ImPts2Dir::Appli_ImPts2Dir(int argc,char ** argv) :
    mOut        ("DirectionsPerImage.xml"),
    mICNM       (0),
    mSetIm      (0)
{
    std::string              aPattern;
    std::vector<std::string> aCircV = {"100","500"};//not implemented for now

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aPattern,"Pattern of images")
                   << EAMC(mOri,"Orientation directory"),
        LArgMain() << EAM (aCircV,"Circ",true,"3-element vector [start_radius,end_radius,interval] en pix")
                   << EAM (mOut,"Out",true,"Output file name")
	);

#if (ELISE_windows)
      replace( aPattern.begin(), aPattern.end(), '\\', '/' );
#endif
    SplitDirAndFile(mDir,mIms,aPattern);
    StdCorrecNameOrient(mOri,mDir);

    mICNM =  cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mSetIm = mICNM->Get(mIms);
    mNbIm = (int)mSetIm->size();



    //verify input
    ELISE_ASSERT(aCircV.size()==3,"Circ vector size must equal 3 in Appli_ImPts2Dir::Appli_ImPts2Dir");


    int aCirc0 = RequireFromString<double>(aCircV.at(0),"Rad start");
    int aCirc1 = RequireFromString<double>(aCircV.at(1),"Rad end");
    int aCirc2 = RequireFromString<double>(aCircV.at(2),"Rad int");

    int aRad0   = aCirc0 > aCirc1 ? aCirc1 : aCirc0;
    int aRadEnd = aCirc0 > aCirc1 ? aCirc0 : aCirc1;
    int aRadInc = aCirc2;
    

    //circles wrt to "0"
    for (int aK=aRad0; aK<=aRadEnd; aK+=aRadInc)
    {
        cFastCriterCompute * aCircRI = cFastCriterCompute::Circle(aK);
        const  std::vector<Pt2di> & aVPt = aCircRI->VPt();

        for ( auto aFlux : aVPt )
            mListPt2d.push_back(Pt2dr(aFlux.x,aFlux.y));

        
    }
/*
    for ( auto aCircRad : aCircV )
    {
        double aRadCur = RequireFromString<double>(aCircRad,"Radius i");

        cFastCriterCompute * aCircRI = cFastCriterCompute::Circle(aRadCur);
        const  std::vector<Pt2di> & aVPt = aCircRI->VPt();

        for ( auto aFlux : aVPt )
            mListPt2d.push_back(Pt2dr(aFlux.x,aFlux.y));
        
    }*/
    mNbPts = int(mListPt2d.size());



}

int Appli_ImPts2Dir::DoCalc()
{
    for (int aKIm=0; aKIm<mNbIm; aKIm++)
    {
        const std::string aNameIm = (*mSetIm)[aKIm];
        
        cBasicGeomCap3D * aCG =  mICNM->StdCamGenerikOfNames(mOri,aNameIm);
        
        mMapImDirs[aNameIm] = new cImDir(aNameIm);
      
        //optical center 
        CamStenope * aCam = aCG->DownCastCS();
        mMapImDirs[aNameIm]->OC() = aCam->Capteur2RayTer(aCam->PP()); 
        mMapImDirs[aNameIm]->PP() = aCam->PP(); 
        

        for (auto aKP : mListPt2d)
        {
    //       std::cout << aKP + mMapImDirs[aNameIm]->PP()  << " 0.0 " << "\n"; 
           ElSeg3D aDir = aCG->Capteur2RayTer(aKP + mMapImDirs[aNameIm]->PP());
       
           mMapImDirs[aNameIm]->mDirs.push_back(aDir);     
        } 
        

    }

    Save();



    return EXIT_SUCCESS;
}

int Appli_ImPts2Dir::Save()
{

    cXml_ImSetDir aXmlSet;

    std::list< cXml_ImDir > aXmlDirList;
    for (int aKIm=0; aKIm<mNbIm; aKIm++)
    {
        const std::string aNameIm = (*mSetIm)[aKIm];

        cXml_ImDir aXmlIm;
        aXmlIm.Name() = aNameIm;
        aXmlIm.P1OC() = mMapImDirs[aNameIm]->OC().P0();
        aXmlIm.P2OC() = mMapImDirs[aNameIm]->OC().P1();

        std::list< cXml_SingleDir > aXmlList;
        for (int aP=0; aP<mNbPts; aP++)
        {
            cXml_SingleDir aXmlDir;
            aXmlDir.PIm() = mListPt2d.at(aP) + mMapImDirs[aNameIm]->PP();
            aXmlDir.P1() = mMapImDirs[aNameIm]->mDirs.at(aP).P0();
            aXmlDir.P2() = mMapImDirs[aNameIm]->mDirs.at(aP).P1();

            aXmlList.push_back(aXmlDir);
        }
        aXmlIm.ListDir() = aXmlList;

        aXmlDirList.push_back(aXmlIm);


    }

    aXmlSet.Ims() = aXmlDirList;
    MakeFileXML(aXmlSet,mOut);

    return EXIT_SUCCESS;
}

/* Extraction of direction for Manchun */
int ImPts2Dir_main(int argc,char ** argv)
{
    Appli_ImPts2Dir aAppDir(argc,argv);


    if (aAppDir.DoCalc())
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;


}


void TestEllips_3D();
int  FictiveObstest_main(int argc,char ** argv)
{
    if (0)
    {
        cPlyCloud aPlyEl0, aPlyEl1;

        cXml_Elips2D  anEl;
        RazEllips(anEl);

        //P1
        Pt2dr aP1(-10,20);
        AddEllips(anEl,aP1,1.0);
        //P2
        Pt2dr aP2(10,20); 
        AddEllips(anEl,aP2,1.0);
        //P3
        Pt2dr aP3(-10,-20);
        AddEllips(anEl,aP3,1.0);
        //P4
        Pt2dr aP4(10,20);
        AddEllips(anEl,aP4,1.0);
        //P5
        Pt2dr aP5(0,0);
        AddEllips(anEl,aP5,1.0);
        //P6 outlier
        Pt2dr aP6(0,100);
        AddEllips(anEl,aP6,1.0);


        NormEllips(anEl);

        cGenGaus2D aGG1(anEl);
        std::vector<Pt2dr> aVP;

        aGG1.GetDistr3Points(aVP);
	aGG1.GetDistribGaus(aVP,1,1);

        //save
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP1.x,aP1.y,0));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP2.x,aP2.y,0));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP3.x,aP3.y,0));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP4.x,aP4.y,0));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP5.x,aP5.y,0));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP6.x,aP6.y,0));
        aPlyEl0.PutFile("PtsOrg.ply");
        
        for (int aK=0 ; aK<int(aVP.size()) ; aK++)
            aPlyEl1.AddPt(Pt3di(0,0,255),Pt3dr(aVP[aK].x,aVP[aK].y,0));

        aPlyEl1.PutFile("ElGen.ply");


    }
	
    if (1)
    {
        cPlyCloud aPlyEl0, aPlyEl1;

        cXml_Elips3D  anEl;
        RazEllips(anEl);

		        //P1
        Pt3dr aP1(-10,20,0);
        AddEllips(anEl,aP1,1.0);
        //P2
        Pt3dr aP2(10,20,-5);
        AddEllips(anEl,aP2,1.0);
        //P3
        Pt3dr aP3(-10,-20,5);
        AddEllips(anEl,aP3,1.0);
        //P4
        Pt3dr aP4(10,20,0);
        AddEllips(anEl,aP4,1.0);
        //P5
        Pt3dr aP5(0,0,5);
        AddEllips(anEl,aP5,1.0);
        //P6 outlier
        Pt3dr aP6(0,100,-5);
        AddEllips(anEl,aP6,1.0);


        NormEllips(anEl);

        cGenGaus3D aGG1(anEl);
        std::vector<Pt3dr> aVP;
	
		aGG1.GetDistr5Points(aVP);

		//save
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP1.x,aP1.y,aP1.z));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP2.x,aP2.y,aP2.z));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP3.x,aP3.y,aP3.z));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP4.x,aP4.y,aP4.z));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP5.x,aP5.y,aP5.z));
        aPlyEl0.AddPt(Pt3di(255,0,0),Pt3dr(aP6.x,aP6.y,aP6.z));
        aPlyEl0.PutFile("PtsOrg-3D.ply");

        for (int aK=0 ; aK<int(aVP.size()) ; aK++)
            aPlyEl1.AddPt(Pt3di(0,0,255),Pt3dr(aVP[aK].x,aVP[aK].y,aVP[aK].z));

        aPlyEl1.PutFile("ElGen5Pts-3D.ply");

		
    }


    /*  (0) add all points to an ellipse classe cXml_Elips3D 
        (1) normalise the ellipse
        (2) initalize the cGenGaus3D with the ellipse (+calcul valp/vecp)
        (3) generate artif points (aVP vec) that follow the distribution 
            of the ellipse (eigenvalues) 
        (4) add the aVP points again the the ellipse (after reset) + norm
        (5) initialize a new cGenGaus3D with the new ellipse 
        (6) check the ratio of VP for three dimensions */
    while (0)
    {
        cPlyCloud aPlyEl0, aPlyEl1;


        int aNbPts =  4 + NRrandom3(20);
        cXml_Elips3D  anEl;

        Pt3dr aC0 (NRrandC(),NRrandC(),NRrandC());
        Pt3dr aU0 (NRrandC(),NRrandC(),NRrandC());
        Pt3dr aU1 (NRrandC(),NRrandC(),NRrandC());
        Pt3dr aU2 (NRrandC(),NRrandC(),NRrandC());
        RazEllips(anEl);
        for (int aK=0 ; aK<aNbPts ; aK++)
        {
             // Pt3dr aP NRrandC(),NRrandC(),NRrandC());
             Pt3dr aP = aC0 + aU0 *  NRrandC() + aU1 * NRrandC() + aU2 * NRrandC();
             aP = aP * 10;
             AddEllips(anEl,aP,1.0);
             if(1)
                aPlyEl0.AddPt(Pt3di(255,255,255),aP);
        }
        NormEllips(anEl);


        cGenGaus3D aGG1(anEl);
        std::vector<Pt3dr> aVP;

        aGG1.GetDistribGaus(aVP,1+NRrandom3(2),2+NRrandom3(2),3+NRrandom3(2));
    
        if(1)
            for (int aK=0 ; aK<int(aVP.size()) ; aK++)
                aPlyEl1.AddPt(Pt3di(255,255,255),aVP[aK]);

        RazEllips(anEl);
        for (int aK=0 ; aK<int(aVP.size()) ; aK++)
            AddEllips(anEl,aVP[aK],1.0);

        NormEllips(anEl);
        cGenGaus3D aGG2(anEl);

        for (int aK=0 ; aK< 3 ; aK++)
        {
            std::cout << "RATIO VP " << aGG1.ValP(aK) /  aGG2.ValP(aK) << " "
                      <<  euclid( aGG1.VecP(aK) - aGG2.VecP(aK))       << " "
                      <<  " VP="  << aGG1.ValP(aK)       << " "
                      << "\n";
        }

        aPlyEl0.PutFile("El0-" + ToString(aNbPts) + ".ply");
        aPlyEl1.PutFile("El1-" + ToString(aNbPts) + ".ply");

        getchar();
    }
        

    return EXIT_SUCCESS;
}


int Test_Homogr_main(int argc, char ** argv)
{
    std::string aIn="";
    Pt2dr aP1, aP2;
    ElPackHomologue aPack;


    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aIn,"2D correspondences"),
        LArgMain() 
    );

#if (ELISE_windows)
      replace( aIn.begin(), aIn.end(), '\\', '/' );
#endif

    ELISE_fp aFIn(aIn.c_str(),ELISE_fp::READ);
    char * aLine;

    while ((aLine = aFIn.std_fgets()))
    {
         int aNb=sscanf(aLine,"%lf  %lf %lf  %lf",&aP1.x , &aP1.y, &aP2.x , &aP2.y);
         ELISE_ASSERT(aNb==4,"Could not read 2 double values");

         std::cout << aP1 << " " << aP2 << "\n";
         ElCplePtsHomologues aP(aP1,aP2);
    
         aPack.Cple_Add(aP);

    }
        
    cElHomographie  aHom = cElHomographie::RansacInitH(aPack,50,2000);
    cElHomographie  aHomInv = aHom.Inverse();
    aHom.Show();
    aHomInv.Show();

    Pt2dr aTP1(1589.52418648700177,1214.28114830960817);
    Pt2dr aTP2(8317.05902029607387,555.169478782111128);

    std::cout << aHom.Direct(aTP1) << " " << aHomInv.Direct(aTP2) << "  " << "\n";

    cElComposHomographie aHX = aHom.HX();
    //cElComposHomographie aHY = aHom.HY();
    cElComposHomographie aHZ = aHom.HZ();
    double aX2  = aTP1.x*aHX.CoeffX() + aTP1.y*aHX.CoeffY() + aHX.Coeff1(); 
    double aDiv = aTP1.x*aHZ.CoeffX() + aTP1.y*aHZ.CoeffY() + aHZ.Coeff1(); 

    std::cout << aTP2.x << " " << aX2 << " " << aX2 / aDiv << "\n";
/*    
*/


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

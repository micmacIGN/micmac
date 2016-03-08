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

   /* CameraRPC * aRPC = new CameraRPC(aFullName, aType);
    cComp3DBasic * aRPCB = new cComp3DBasic (aRPC);
    
    Pt2dr aP2d(100,500);
    ElSeg3D aElOrg = aRPC->Capteur2RayTer(aP2d);
    ElSeg3D aElGeo = aRPCB->Capteur2RayTer(aP2d);

    Pt3dr aPB1 = aRPCB->Target2OriginCS(aElGeo.P0());
    Pt3dr aPB2 = aRPCB->Target2OriginCS(aElGeo.P1());

    std::cout <<  "aElOrg " << aElOrg.P0() << " " 
	                    << aElOrg.P1() << "\n";
    std::cout <<  "aElGeo " << aElGeo.P0() << " " 
	                    << aElGeo.P1() << "\n";
    std::cout <<  "aElGeoBack " << aPB1 << " " 
	                        << aPB2 << "\n";

    //aRPC.OptiicalCenterPerLine();

    //Pt3dr aP1, aP2, aP3;
    //aP1 = aRPC.OpticalCenterOfPixel(Pt2dr(1,1));
    //aP2 = aRPC.OpticalCenterOfPixel(Pt2dr(10,10));
    //aP3 = aRPC.OpticalCenterOfPixel(Pt2dr(aRPC.SzBasicCapt3D().x-1,
//			             aRPC.SzBasicCapt3D().y-1));

  //  std::cout <<  aP1.x << " " << aP1.y << " " << aP1.z << "\n";
  //  std::cout <<  aP2.x << " " << aP2.y << " " << aP2.z << "\n";
  //  std::cout <<  aP3.x << " " << aP3.y << " " << aP3.z << "\n";
*/
    return EXIT_SUCCESS;
}

//test camera affine
int TestER_main3(int argc,char ** argv)
{
    //cInterfChantierNameManipulateur * aICNM;
    std::string aFullName;
    std::string aDir;
    std::string aNameOri;
    std::list<std::string> aListFile;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName,"Orientation file full name (Dir+OriPattern)"),
	LArgMain()
    );

    std::cout << aFullName << std::endl;

    //CameraAffine aCamAF(aFullName);
    //aCamAF.ShowInfo();

    return EXIT_SUCCESS;
}
//test export of a CamStenope into bundles of rays
int TestER_main2(int argc,char ** argv)
{
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

int TestER_rpc(int argc,char ** argv)
{
    std::string aFullName;
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName,"Orientation file"),
        LArgMain()
    );

    cRPC aRPC(aFullName);
    aRPC.Show();
    
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

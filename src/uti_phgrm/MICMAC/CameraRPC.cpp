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

#include "CameraRPC.h"

/***********************************************************************/
/*                           CameraRPC                                 */
/***********************************************************************/

/* Image coordinates order: [Line, Sample] = [row, col] =  [x, y]*/

CameraRPC::CameraRPC(std::string const &aNameFile, const  std::string &aModeRPC, const Pt2di &aGridSz, std::string const &aMetaFile) :
       mProfondeurIsDef(false),	
       mAltisSolIsDef(false),
       mOptCentersIsDef(false),
       mGridSzIsDef(true),
       mOpticalCenters(0),
       mGridSz(aGridSz),
       mCamNom(aNameFile.substr(0,aNameFile.size()-4))
{
   
   mRPC = new RPC();
   
   if (aModeRPC=="PLEIADE")
   {
       mRPC->ReadDimap(aNameFile);
       mRPC->ReconstructValidity();
   }
   else if(aModeRPC=="SPOT")//OK for SPOT in Dimap2 format, otherwise CameraAffine
   {
       mRPC->ReadDimap(aNameFile);   
       mRPC->ReconstructValidity(); 
   }
   else if(aModeRPC=="QUICKBIRD" || aModeRPC=="WORLDVIEW" )
   {
       mRPC->ReadXML(aNameFile);
       mRPC->InverseToDirectRPC(Pt2di(50,50));
   }
   else if(aModeRPC=="IKONOS" || aModeRPC=="CARTOSAT")
   {
       mRPC->ReadASCII(aNameFile);
       mRPC->ReadASCIIMetaData(aMetaFile, aNameFile);
       mRPC->InverseToDirectRPC(Pt2di(50,50));
   }
   else {ELISE_ASSERT(false,"Unknown RPC mode");}

  mRPC->info();


}
CameraRPC::~CameraRPC()
{
	delete mRPC;
	delete mOpticalCenters;
}

Pt2dr CameraRPC::Ter2Capteur(const Pt3dr & aP) const
{
    AssertRPCInvInit();

    Pt3dr aPIm = mRPC->InverseRPC(aP);    
    return Pt2dr(aPIm.x, aPIm.y);
}

bool CameraRPC::PIsVisibleInImage   (const Pt3dr & aP) const
{
    //Project 3D-2D with RPC and see if within ImSz
    Pt2di aSz = SzBasicCapt3D(); 

    Pt2dr aPtProj = Ter2Capteur(aP);

    if( (aPtProj.x >= 0) &&
        (aPtProj.x < aSz.x) &&
	(aPtProj.y >= 0) &&
	(aPtProj.y < aSz.y) )
    	return  true;
    else
	return  false;
}

ElSeg3D  CameraRPC::Capteur2RayTer(const Pt2dr & aP) const
{
    AssertRPCDirInit(); 

    double aZ = mRPC->first_height+1;//beginning of the height validity zone
    if(AltisSolIsDef())
        aZ = mAltiSol;
    
    Pt3dr aP1RayL3(aP.x, aP.y, 
		   aZ+double(mRPC->last_height - mRPC->first_height)/2),//middle of the height validity zones 
	  aP2RayL3(aP.x, aP.y, aZ);

    return F2toRayonLPH(aP1RayL3, aP2RayL3);
}

ElSeg3D CameraRPC::F2toRayonLPH(Pt3dr &aP0,Pt3dr & aP1) const
{
    return(ElSeg3D(mRPC->DirectRPC(aP0), mRPC->DirectRPC(aP1)));
}

bool   CameraRPC::HasRoughCapteur2Terrain() const
{
    return ProfIsDef();
}

Pt3dr CameraRPC::RoughCapteur2Terrain(const Pt2dr & aP) const
{
    if (ProfIsDef())
        return (ImEtProf2Terrain(aP, GetProfondeur()));
    
    if (AltisSolIsDef())
	return(ImEtZ2Terrain(aP, GetAltiSol()));
    
    ELISE_ASSERT(false,"Nor Alti, nor prof : Camera has no \"RoughCapteur2Terrain\"  functionality");
    return(Pt3dr(0,0,0));        
}

Pt3dr CameraRPC::ImEtProf2Terrain(const Pt2dr & aP,double aProf) const
{
    AssertRPCDirInit();

    if(mOptCentersIsDef)
    {
        //find the  sensor's mean Z-position
	unsigned int aK;
	double aMeanZ(mOpticalCenters->at(0).z);
	for(aK=1; aK<mOpticalCenters->size(); aK++)
	    aMeanZ += mOpticalCenters->at(aK).z;

	aMeanZ = double(aMeanZ)/mOpticalCenters->size();

        return(mRPC->InverseRPC(Pt3dr(aP.x, aP.y, aMeanZ-mProfondeur)));
    }
    else
    {
        ELISE_ASSERT(false,"CameraRPC::ImEtProf2Terrain no data about the sensor positon");
	
	return(Pt3dr(0,0,0));
    }
}

Pt3dr CameraRPC::ImEtZ2Terrain(const Pt2dr & aP,double aZ) const
{
    AssertRPCDirInit();

    return(mRPC->InverseRPC(Pt3dr(aP.x, aP.y, aZ)));
}

void CameraRPC::SetProfondeur(double aP)
{
    mProfondeur = aP;
    mProfondeurIsDef = true;
}

double CameraRPC::GetProfondeur() const
{
    return(mProfondeur);
}

bool CameraRPC::ProfIsDef() const
{
    return(mProfondeurIsDef);
}

void CameraRPC::SetAltiSol(double aZ)
{
    mAltiSol = aZ;
    mAltisSolIsDef = true;
}

double CameraRPC::GetAltiSol() const
{
    return(mAltiSol);
}

bool CameraRPC::AltisSolIsDef() const
{
    return(mAltisSolIsDef);
}

double CameraRPC::ResolSolOfPt(const Pt3dr & aP) const
{
    //to do
    return 1.0;//aSeg.DistDoite(aP);
}

bool  CameraRPC::CaptHasData(const Pt2dr & aP) const
{
    //to do
    return true;//IndexHasContenuForInterpol(aP);
}

void CameraRPC::AssertRPCDirInit() const
{
    ELISE_ASSERT(mRPC->IS_DIR_INI,"CameraRPC::AssertRPCDirInit");
}

void CameraRPC::AssertRPCInvInit() const
{
    ELISE_ASSERT(mRPC->IS_INV_INI,"CameraRPC::AssertRPCInvInit");
}

Pt2di CameraRPC::SzBasicCapt3D() const
{
    ELISE_ASSERT(mRPC!=0,"RPCs were not initialized in CameraRPC::SzBasicCapt3D()");

    return  (Pt2di(mRPC->last_row,
 	           mRPC->last_col));
}

/* Export to xml following the cXml_ScanLineSensor standard 
 * - first  iter - generate the bundle grid in geodetic coordinate system (CS) and
 *                 convert to desired CS
 * - second iter - export to xml */
void CameraRPC::ExpImp2Bundle(const std::string & aSysOut, 
			      std::vector<std::vector<ElSeg3D> > aGridToExp) const
{
        //Check that the direct RPC exists
	AssertRPCDirInit();

	Pt2dr aGridStep = Pt2dr( double(SzBasicCapt3D().x)/mGridSz.x ,
			         double(SzBasicCapt3D().y)/mGridSz.y );

	std::string aDirTmp = "csconv";
	std::string aFiPrefix = "Bundle_";

	std::string aLPHFiTmp = aDirTmp + "/" + aFiPrefix + mCamNom  + "_LPH_CS.txt";
	std::string aXYZFiTmp = aDirTmp + "/" + aFiPrefix + mCamNom  + "_XYZ_CS.txt";
	std::string aXMLFiTmp = aFiPrefix + mCamNom  + ".xml";

	int aL=0, aS=0;
	if(aGridToExp.size()==0)
	{
		ELISE_fp::MkDirSvp(aDirTmp);
		std::ofstream aFO(aLPHFiTmp.c_str());
		aFO << std::setprecision(15);
	
		//create the bundle grid in geodetic CS & save	
		ElSeg3D aSegTmp(Pt3dr(0,0,0),Pt3dr(0,0,0));
		for( aL=0; aL<mGridSz.x; aL++ )
			for( aS=0; aS<mGridSz.y; aS++ )
			{

				aSegTmp = Capteur2RayTer( Pt2dr(aS*aGridStep.y,aL*aGridStep.x) );
				aFO << aSegTmp.P0().x << " " << aSegTmp.P0().y << " " << aSegTmp.P0().z << "\n" 
				    << aSegTmp.P1().x << " " << aSegTmp.P1().y << " " << aSegTmp.P1().z << "\n";

			}
		aFO.close();

		//convert from goedetic CS to the user-selected CS 
		std::string aCmdTmp = " " + aLPHFiTmp + " > " + aXYZFiTmp;
		std::string cmdConv = g_externalToolHandler.get("cs2cs").callName() + " " + 
			             "+proj=longlat +datum=WGS84" + " +to " + aSysOut + 
				     aCmdTmp;

		int aRes = system(cmdConv.c_str());		
		ELISE_ASSERT(aRes == 0, " Error calling cs2cs");

		//read-in the converted bundle grid
		std::vector<Pt3dr> aPtsTmp;
		double aXtmp, aYtmp, aZtmp;
		std::ifstream aFI(aXYZFiTmp.c_str());
		while( !aFI.eof() && aFI.good() )
		{
			aFI >> aXtmp >> aYtmp >> aZtmp;
			aPtsTmp.push_back(Pt3dr(aXtmp,aYtmp,aZtmp));

		}
		aFI.close();

		aGridToExp.resize(mGridSz.y);
		int aCntTmp=0;
		for( aL=0; aL<mGridSz.x; aL++ )
			for( aS=0; aS<mGridSz.y; aS++ )
			{
				aGridToExp.at(aL).push_back ( ElSeg3D(aPtsTmp.at(aCntTmp), 
							              aPtsTmp.at(aCntTmp+1)) );
				aCntTmp++;
				aCntTmp++;

			}

		ExpImp2Bundle(aSysOut, aGridToExp);
	}
	else
	{

		cXml_ScanLineSensor aSLS;

		aSLS.P1P2IsAltitude() = HasRoughCapteur2Terrain();
		aSLS.LineImIsScanLine() = true;
		aSLS.GroundSystemIsEuclid() = true;    

		aSLS.ImSz() = SzBasicCapt3D();

		aSLS.StepGrid() = aGridStep;

		aSLS.GridSz() = mGridSz;	

		for( aL=0; aL<mGridSz.x; aL++ )
		{
			cXml_OneLineSLS aOL;
			aOL.IndLine() = aL*aGridStep.x;
			for( aS=0; aS<mGridSz.y; aS++ )
			{
				cXml_SLSRay aOR;
				aOR.IndCol() = aS*aGridStep.y;

				aOR.P1() = aGridToExp.at(aL).at(aS).P0();
				aOR.P2() = aGridToExp.at(aL).at(aS).P1();

				aOL.Rays().push_back(aOR);

			}
			aSLS.Lines().push_back(aOL);
		}
		//export to XML format
		MakeFileXML(aSLS, aXMLFiTmp);
	}		    
}

/* For a defined image grid, 
 * extrude to rays and intersect at the line of optical centers */
void CameraRPC::OpticalCenterLineTer(const std::string & aCSysOut, bool aIfSave)
{
    int aL, aS;
    std::vector<Pt3dr> aVPts;
    std::string aDirTmp = "PBoptCenter";
    std::string aSavGeo = aDirTmp + "/bundleGridGeo" + mCamNom + ".txt";
    std::string aSavUCS = aDirTmp + "/bundleGridUCS" + mCamNom + ".txt";
    std::string aSaveFile = aDirTmp + "/OpticalCentersTer" + mCamNom + ".txt";
    

    //define a default grid size unless previously defined
    if(!mGridSzIsDef)
        mGridSz = Pt2di(10,10);

    Pt2dr aGridStep = Pt2dr( double(SzBasicCapt3D().x)/mGridSz.x ,
                             double(SzBasicCapt3D().y)/mGridSz.y );



    //collect the bundles in geodetic CS and save to txt
    ELISE_fp::MkDirSvp(aDirTmp);
    std::ofstream aFO(aSavGeo.c_str());
    aFO << std::setprecision(15);

    ElSeg3D aSegTmp(Pt3dr(0,0,0), Pt3dr(0,0,0));
    for( aL=0; aL<mGridSz.x; aL++)
    {
	for( aS=0; aS<mGridSz.y; aS++)
	{
	    //std::cout << aS*aGridStep.y << " " << aL*aGridStep.x << " - ";
	    aSegTmp = Capteur2RayTer( Pt2dr( aS*aGridStep.y, aL*aGridStep.x));
            aFO << aSegTmp.P0().x << " " << aSegTmp.P0().y << " " << aSegTmp.P0().z << "\n"   
                << aSegTmp.P1().x << " " << aSegTmp.P1().y << " " << aSegTmp.P1().z << "\n";
	}
	//std::cout << "\n";
    }
    aFO.close();



    //convert from geodetic CS to the user-defined CS
    std::string aCmdTmp = " " + aSavGeo + " > " + aSavUCS;
    std::string cmdConv = g_externalToolHandler.get("cs2cs").callName() + " " +
                         "+proj=longlat +datum=WGS84" + " +to " + aCSysOut + aCmdTmp;
    int aRes = system(cmdConv.c_str());
    ELISE_ASSERT(aRes == 0, " Error calling cs2cs");


    //read-in the converted bundles
    std::vector<Pt3dr> aPtsTmp;
    double aXtmp, aYtmp, aZtmp;
    std::ifstream aFI(aSavUCS.c_str());
    while( !aFI.eof() && aFI.good() )
    {
        aFI >> aXtmp >> aYtmp >> aZtmp;
        aPtsTmp.push_back(Pt3dr(aXtmp,aYtmp,aZtmp));
    }
    aFI.close();


    //do the intersection on respective segments in the user-def CS
    mOpticalCenters = new std::vector<Pt3dr>();
    //cRapOnZ * aRAZ = new cRapOnZ(680000.0,10,10,"");
    //cResOptInterFaisceaux * aROIF;
    std::cout.precision(15);
    int aCntTmp=0;
    for( aL=0; aL<mGridSz.x; aL++)
    {
	std::vector<ElSeg3D> aVS;
	std::vector<double>  aVPds;

        
	for( aS=0; aS<mGridSz.y; aS++)
	{
	    aVPds.push_back(0.5);
            aVS.push_back( ElSeg3D(aPtsTmp.at(aCntTmp),
				   aPtsTmp.at(aCntTmp+1)) );

	   // std::cout << aPtsTmp.at(aCntTmp) << " " << aPtsTmp.at(aCntTmp+1) << " - ";
	    aCntTmp++;
	    aCntTmp++;

	}
	//std::cout << "\n";

	//intersect
	bool aIsOK;
	//mOpticalCenters->push_back( ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK, aRAZ) );
	mOpticalCenters->push_back( ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK) );



	if(aIsOK==false)
	    std::cout << "not intersected in CameraRPC::OpticalCenterLineTer" << std::endl;

    }
    mOptCentersIsDef = true; 

    if(aIfSave)
    {
        std::ofstream aFO(aSaveFile.c_str());
        aFO << std::setprecision(15);

        unsigned int i;	
	for(i=0; i<mOpticalCenters->size(); i++)
	    aFO << mOpticalCenters->at(i).x 
		<< " " << mOpticalCenters->at(i).y 
		<< " " << mOpticalCenters->at(i).z
		<< "\n";
	
        aFO.close();

    }


}


/***********************************************************************/
/*                           CameraAffine                              */
/***********************************************************************/
CameraAffine::CameraAffine(std::string const &file)
{
   cElXMLTree tree(file.c_str()); 
   cElXMLTree* nodes;
   std::list<cElXMLTree*> nodesFilAll;

   //camera affine parameters
   nodes = tree.GetUnique(std::string("Direct_Location_Model"));

   nodesFilAll = nodes->GetAll("lc");
   std::list<cElXMLTree*>::const_iterator aK;
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
      mCDir_LON.push_back(std::atof((*aK)->Contenu().c_str()));

   nodesFilAll = nodes->GetAll("pc");
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
       mCDir_LAT.push_back(std::atof((*aK)->Contenu().c_str()));


   nodes = tree.GetUnique(std::string("Reverse_Location_Model"));

   nodesFilAll = nodes->GetAll("lc");
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
       mCInv_Line.push_back(std::atof((*aK)->Contenu().c_str()));

   nodesFilAll = nodes->GetAll("pc");
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
       mCInv_Sample.push_back(std::atof((*aK)->Contenu().c_str()));


   //validity zones
   std::vector<double> aValTmp;
   nodes = tree.GetUnique(std::string("Dataset_Frame"));
   nodesFilAll = nodes->GetAll("FRAME_LON");
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
       aValTmp.push_back(std::atof((*aK)->Contenu().c_str()));

   mLON0 = (*std::min_element(aValTmp.begin(), aValTmp.end()));
   mLONn = (*std::max_element(aValTmp.begin(), aValTmp.end()));
   aValTmp.clear();   

   nodesFilAll = nodes->GetAll("FRAME_LAT");
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
       aValTmp.push_back(std::atof((*aK)->Contenu().c_str()));

   mLAT0 = (*std::min_element(aValTmp.begin(), aValTmp.end()));
   mLATn = (*std::max_element(aValTmp.begin(), aValTmp.end()));
   aValTmp.clear();

   //row
   nodesFilAll = nodes->GetAll("FRAME_ROW");
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
        aValTmp.push_back(std::atof((*aK)->Contenu().c_str()));

   mROW0 = (*std::min_element(aValTmp.begin(), aValTmp.end()));
   mROWn = (*std::max_element(aValTmp.begin(), aValTmp.end()));
   aValTmp.clear();

   //col
   nodesFilAll = nodes->GetAll("FRAME_COL");
   for( aK=nodesFilAll.begin(); aK!=nodesFilAll.end(); aK++ )
        aValTmp.push_back(std::atof((*aK)->Contenu().c_str()));

   mCOL0 = (*std::min_element(aValTmp.begin(), aValTmp.end()));
   mCOLn = (*std::max_element(aValTmp.begin(), aValTmp.end()));
   aValTmp.clear();

   //sensor size
   nodes = tree.GetUnique("NROWS");
   mSz.x = std::atoi(nodes->GetUniqueVal().c_str());
   nodes = tree.GetUnique("NCOLS");
   mSz.y = std::atoi(nodes->GetUniqueVal().c_str());

}

ElSeg3D CameraAffine::Capteur2RayTer(const Pt2dr & aP) const
{
    return( ElSeg3D(Pt3dr(0,0,0), Pt3dr(0,0,0)) );
}

Pt2dr CameraAffine::Ter2Capteur   (const Pt3dr & aP) const
{
    return(Pt2dr(0,0));
}

Pt2di CameraAffine::SzBasicCapt3D() const
{
    return(mSz);
}

double CameraAffine::ResolSolOfPt(const Pt3dr &) const
{
    return( 0.0 );
}

bool CameraAffine::CaptHasData(const Pt2dr &) const
{
    return(true);
}

bool CameraAffine::PIsVisibleInImage   (const Pt3dr & aP) const
{
    return(true);
}

Pt3dr CameraAffine::OpticalCenterOfPixel(const Pt2dr & aP) const
{
    return(Pt3dr(0,0,0));
}

bool  CameraAffine::HasOpticalCenterOfPixel() const
{
    return(true);
}

void CameraAffine::Diff(Pt2dr & aDx,Pt2dr & aDy,Pt2dr & aDz,const Pt2dr & aPIm,const Pt3dr & aTer)
{}

void CameraAffine::ShowInfo()
{
    unsigned int aK=0;

    std::cout << "CameraAffine info:" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "------------------direct model\n";
    std::cout << "lambda = ";
    for(aK=0; aK<mCDir_LON.size(); aK++)
	    std::cout << "(" << aK << ") " << mCDir_LON.at(aK) << "\n ";

    std::cout << "\n";

    std::cout << "phi = ";
    for(aK=0; aK<mCDir_LAT.size(); aK++)
	    std::cout << "(" << aK << ") " << mCDir_LAT.at(aK) << "\n ";

    std::cout << "\n------------------inverse model\n";
    std::cout << "line = ";
    for(aK=0; aK<mCInv_Line.size(); aK++)
	    std::cout << "(" << aK << ") " << mCInv_Line.at(aK) << "\n ";

    std::cout << "\n";

    std::cout << "line = ";
    for(aK=0; aK<mCInv_Sample.size(); aK++)
	    std::cout << "(" << aK << ") " << mCInv_Sample.at(aK) << "\n ";

    std::cout << "\n------------------validity zones\n";
    std::cout << "LAT = (" << mLAT0 << " - " << mLATn << ")\n";
    std::cout << "LON = (" << mLON0 << " - " << mLONn << ")\n";
    std::cout << "ROW = (" << mROW0 << " - " << mROWn << ")\n";
    std::cout << "COL = (" << mCOL0 << " - " << mCOLn << ")\n";

    std::cout << "\n------------------image dimension\n";
    std::cout << "(nrows,ncols) = (" << mSz.x << ", " << mSz.y << ")\n";
    std::cout << "=================================================" << std::endl;
}

/***********************************************************************/
/*                        BundleCameraRPC                              */
/***********************************************************************/

BundleCameraRPC::BundleCameraRPC(cCapture3D * aCam) : mCam(aCam)
{}

Pt2dr BundleCameraRPC::Ter2Capteur(const Pt3dr & aP) const
{
	return(mCam->Ter2Capteur(aP));
}

bool BundleCameraRPC::PIsVisibleInImage(const Pt3dr & aP) const
{
	return(mCam->PIsVisibleInImage(aP));
}

ElSeg3D BundleCameraRPC::Capteur2RayTer(const Pt2dr & aP) const
{
	return(mCam->Capteur2RayTer(aP));
}
Pt2di BundleCameraRPC::SzBasicCapt3D() const
{
	return(mCam->SzBasicCapt3D());
}

bool  BundleCameraRPC::HasRoughCapteur2Terrain() const
{
	return(mCam->HasRoughCapteur2Terrain());
}

bool  BundleCameraRPC::HasPreciseCapteur2Terrain() const
{
	return(mCam->HasPreciseCapteur2Terrain());
}

Pt3dr BundleCameraRPC::RoughCapteur2Terrain   (const Pt2dr & aP) const
{
	return(mCam->RoughCapteur2Terrain(aP));
}

Pt3dr BundleCameraRPC::PreciseCapteur2Terrain   (const Pt2dr & aP) const
{
	return(mCam->PreciseCapteur2Terrain(aP));
}

double BundleCameraRPC::ResolSolOfPt(const Pt3dr & aP) const
{
	return(mCam->ResolSolOfPt(aP));
}

double BundleCameraRPC::ResolSolGlob() const
{
	return(mCam->ResolSolGlob());
}

bool  BundleCameraRPC::CaptHasData(const Pt2dr & aP) const
{
	return(mCam->CaptHasData(aP));
}

Pt2dr BundleCameraRPC::ImRef2Capteur   (const Pt2dr & aP) const
{
	return(mCam->ImRef2Capteur(aP));
}

double BundleCameraRPC::ResolImRefFromCapteur() const
{
	return(mCam->ResolImRefFromCapteur());
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

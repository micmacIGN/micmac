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

/* Image coordinates order: [Line, Sample] = [row, col] =  [y, x]*/

//this is an outdated constructor -- will be soon removed 
CameraRPC::CameraRPC(const std::string &aNameFile, 
		     const eTypeImporGenBundle &aType,
		     const std::string &aCartoCS, 
		     const Pt2di &aGridSz, 
		     const std::string &aMetaFile) :
       mProfondeurIsDef(false),	
       mAltisSolIsDef(false),
       mOptCentersIsDef(false),
       mGridSzIsDef(true),
       mOpticalCenters(0),
       mGridSz(aGridSz),
       mImName(aNameFile.substr(0,aNameFile.size()-4)),
       mCS(aCartoCS)
{
   
   mRPC = new RPC();
   
   if (aType==eTIGB_MMDimap2)
   {
       mRPC->ReadDimap(aNameFile);
       mRPC->ReconstructValidity();
   }
   else if(aType==eTIGB_MMDimap1)
   {
       ELISE_ASSERT(false,"No eTIGB_MMDimap1");
   }
   else if(aType==eTIGB_MMDGlobe )
   {
       mRPC->ReadXML(aNameFile);
       mRPC->InverseToDirectRPC(Pt3di(50,50,10));
   }
   else if(aType==eTIGB_MMIkonos)
   {
       mRPC->ReadASCII(aNameFile);
       mRPC->ReadASCIIMetaData(aMetaFile, aNameFile);
       mRPC->InverseToDirectRPC(Pt2di(50,50));
   }
   else if (aType == eTIGB_MMASTER)
   {
	   mRPC->AsterMetaDataXML(aNameFile);
	   std::string aNameIm = StdPrefix(aNameFile) + ".tif";
	   //Find Validity and normalization values
	   mRPC->ComputeNormFactors(0, 8000);//0 and 800 are min and max altitude
	   vector<vector<Pt3dr> > aGridNorm = mRPC->GenerateNormLineOfSightGrid(20,0,8000);
	   mRPC->GCP2Direct(aGridNorm[0], aGridNorm[1]);
	   mRPC->GCP2Inverse(aGridNorm[0], aGridNorm[1]);
   }
   else {ELISE_ASSERT(false,"Unknown RPC mode");}

  mRPC->info();

  //if cartographic coordinate system was not defined
  if(mCS == "")
      FindUTMCS();

}

CameraRPC::CameraRPC(const std::string &aNameFile, 
		     const eTypeImporGenBundle &aType,
		     const std::string aCartoCS) : 
	mProfondeurIsDef(false),
	mAltisSolIsDef(false),
	mOptCentersIsDef(false),
	mGridSzIsDef(false),
	mOpticalCenters(0),
	mGridSz(Pt2di(0,0)),
	mImName(aNameFile.substr(0,aNameFile.size()-4)),
	mCS(aCartoCS)
{
    mRPC = new RPC();

    if(aType==eTIGB_MMDimap2)
    {
        mRPC->ReadDimap(aNameFile);
        mRPC->ReconstructValidity();	
    }
    else if(aType==eTIGB_MMDGlobe)
    {
        mRPC->ReadXML(aNameFile);
	mRPC->InverseToDirectRPC(Pt3di(50,50,10));
    }
    /*else if(aType==eTIGB_MMIkonos)
    {
        mRPC->ReadASCII(aNameFile);
	mRPC->ReadASCIIMetaData(aMetaFile, aNameFile);
	mRPC->InverseToDirectRPC(Pt2di(50,50));
    }*/
    else {ELISE_ASSERT(false,"Unknown RPC mode.");}


    //if cartographic coordinate system was not defined
    if(mCS == "")
        FindUTMCS();

    mRPC->info();
}

CameraRPC * CamRPCOrientGenFromFile(const std::string & aName, const eTypeImporGenBundle aType)
{
    CameraRPC * aRes = new CameraRPC(aName, aType);
    
    return aRes;
}

CameraRPC::~CameraRPC()
{
	delete mRPC;
	delete mOpticalCenters;
}

const RPC & CameraRPC::GetRPC() const
{
    return(*mRPC);
}

const std::string & CameraRPC::GetImName() const
{
    return(mImName);
}

Pt2dr CameraRPC::Ter2Capteur(const Pt3dr & aP) const
{
    AssertRPCInvInit();

    Pt3dr aPIm = mRPC->InverseRPC(aP);    
    return Pt2dr(aPIm.x, aPIm.y);
}

bool CameraRPC::PIsVisibleInImage   (const Pt3dr & aP) const
{
    // (1) Check if aP is within the RPC validity zone
    if( (aP.x < mRPC->first_lon) || (aP.x > mRPC->last_lon) || 
        (aP.y < mRPC->first_lat) || (aP.y > mRPC->last_lat) || 
	(aP.z < mRPC->first_height) || (aP.z > mRPC->last_height) )
        return false;

    // (2) Project 3D-2D with RPC and see if within ImSz
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
    return(ElSeg3D(mRPC->DirectRPC(Pt3dr(aP0.x,aP0.y,aP0.z)), 
	           mRPC->DirectRPC(Pt3dr(aP1.x,aP1.y,aP1.z))));
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

        return(mRPC->DirectRPC(Pt3dr(aP.x, aP.y, aMeanZ-mProfondeur)));
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

    return(mRPC->DirectRPC(Pt3dr(aP.x, aP.y, aZ)));
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
    Pt2dr aPIm = Ter2Capteur(aP);
    ElSeg3D aSeg = Capteur2RayTer(aPIm+Pt2dr(1,0));
    
    return aSeg.DistDoite(aP);
}

bool  CameraRPC::CaptHasData(const Pt2dr & aP) const
{
    if( (aP.y >= mRPC->first_row) && (aP.y <= mRPC->last_row) &&
        (aP.x >= mRPC->first_col) && (aP.x <= mRPC->last_col))
        return true;
    else
	return false;
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

    return  (Pt2di(mRPC->last_col,
 	           mRPC->last_row));
}

/* Export to xml following the cXml_ScanLineSensor standard 
 * - first  iter - generate the bundle grid in geodetic coordinate system (CS) and
 *                 convert to desired CS
 * - second iter - export to xml */
void CameraRPC::ExpImp2Bundle(std::vector<std::vector<ElSeg3D> > aGridToExp) const
{
        //Check that the direct RPC exists
	AssertRPCDirInit();

	Pt2dr aGridStep = Pt2dr( double(SzBasicCapt3D().x)/mGridSz.x ,
			         double(SzBasicCapt3D().y)/mGridSz.y );

	std::string aDirTmp = "csconv";
	std::string aFiPrefix = "Bundle_";

	std::string aLPHFiTmp = aDirTmp + "/" + aFiPrefix + mImName  + "_LPH_CS.txt";
	std::string aXYZFiTmp = aDirTmp + "/" + aFiPrefix + mImName  + "_XYZ_CS.txt";
	std::string aXMLFiTmp = aFiPrefix + mImName  + ".xml";

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

				//aSegTmp = Capteur2RayTer( Pt2dr(aS*aGridStep.y,aL*aGridStep.x) );
				aSegTmp = Capteur2RayTer( Pt2dr(aL*aGridStep.x,aS*aGridStep.y) );
				aFO << aSegTmp.P0().x << " " << aSegTmp.P0().y << " " << aSegTmp.P0().z << "\n" 
				    << aSegTmp.P1().x << " " << aSegTmp.P1().y << " " << aSegTmp.P1().z << "\n";

			}
		aFO.close();

		//convert from goedetic CS to the user-selected CS 
		std::string aCmdTmp = " " + aLPHFiTmp + " > " + aXYZFiTmp;
		std::string cmdConv = g_externalToolHandler.get("cs2cs").callName() + " " + 
			             "+proj=longlat +datum=WGS84" + " +to " + mCS + 
				     aCmdTmp;

		int aRunOK = system(cmdConv.c_str());		
		ELISE_ASSERT(aRunOK == 0, " Error calling cs2cs");

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

		ExpImp2Bundle(aGridToExp);
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

/*
MPD 
void CameraRPC::OpticalCenterrOfImg() const
{
}
*/

/* For a defined image grid, 
 * extrude to rays and intersect at the line of optical centers */
void CameraRPC::OpticalCenterGrid(bool aIfSave) const
{
    srand(time(NULL));

    int aL, aS;
    int aCRand1 = rand() % 255, aCRand2 = rand() % 255, aCRand3 = rand() % 255;
    std::vector<Pt3dr> aVPts;
    std::vector<Pt3di> aVCol;
    Pt3di aCoul(aCRand1,aCRand2,aCRand3);
    std::vector<const cElNuage3DMaille *> aVNuage;

    std::string aDir = "PBoptCenter//";
    std::string aPlyFile = aDir + "OptCenter_" + mImName + ".ply";
    std::list<std::string> aVCom;

    ELISE_fp::MkDirSvp(aDir);

    //define a default grid size unless previously defined
    if(!mGridSzIsDef)
    {
        // MPD : very dirty, just to compile ...
        const_cast<CameraRPC*>(this)->mGridSz = Pt2di(10,10);
    }

    int aAd=1;
    Pt2dr aGridStep = Pt2dr( double(SzBasicCapt3D().x)/(mGridSz.x+aAd) ,
                             double(SzBasicCapt3D().y)/(mGridSz.y+aAd));

    // MPD : very dirty, just to compile ...
    const_cast<CameraRPC*>(this)->mOpticalCenters = new std::vector<Pt3dr>();
    //mOpticalCenters = new std::vector<Pt3dr>();

    Pt3dr aPWGS84, aPGeoC1, aPGeoC2;
    for( aL=aAd; aL<mGridSz.y+aAd; aL++)
    {
        std::vector<double> aVPds;
        std::vector<ElSeg3D> aVS;
	    
	for( aS=aAd; aS<mGridSz.x+aAd; aS++)
	{

            //first height in validity zone
	    aPWGS84 = ImEtZ2Terrain(Pt2dr(aS*aGridStep.x,aL*aGridStep.y),
			                  mRPC->first_height+1);
	    
	    aPGeoC1 = cSysCoord::WGS84()->ToGeoC(aPWGS84);
            aVPts.push_back(aPGeoC1);
            
	    //second height in validity zone
	    aPWGS84 = ImEtZ2Terrain(Pt2dr(aS*aGridStep.x,aL*aGridStep.y), 
			           (mRPC->first_height+1)+double(mRPC->last_height - mRPC->first_height)/2);

	    aPGeoC2 = cSysCoord::WGS84()->ToGeoC(aPWGS84);
	    aVPts.push_back(aPGeoC2);
	    
	    //collect for intersection
	    aVS.push_back(ElSeg3D(aPGeoC1,aPGeoC2));
	    aVPds.push_back(1);

            aVCol.push_back(aCoul);
	    aVCol.push_back(aCoul);

	}

	bool aIsOK;
	mOpticalCenters->push_back( ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK) );

	if(aIsOK==false)
	    std::cout << "not intersected in CameraRPC::OpticalCenterGrid" << std::endl;
    

    }

    //save to ply
    cElNuage3DMaille::PlyPutFile
    (
       aPlyFile,
       aVCom,
       aVNuage,
       mOpticalCenters,
       &aVCol,
       true
    );
}

Pt3dr CameraRPC::OpticalCenterOfPixel(const Pt2dr & aP) const
{
    
    int aS, aAd = 1, aSNum = 10;
    double aSStep = double(SzBasicCapt3D().x)/(mGridSz.x+aAd);
    Pt3dr aPWGS84, aPGeoC1, aPGeoC2;

    std::vector<double> aVPds;
    std::vector<ElSeg3D> aVS;

    for(aS=0; aS<aSNum; aS++)
    {
    
        //first height in validity zone
	aPWGS84 = ImEtZ2Terrain(Pt2dr(aS*aSStep,aP.y),
			        mRPC->first_height+1);
	    
	aPGeoC1 = cSysCoord::WGS84()->ToGeoC(aPWGS84);
            
	//second height in validity zone
	aPWGS84 = ImEtZ2Terrain(Pt2dr(aS*aSStep,aP.y), 
			        (mRPC->first_height+1)+double(mRPC->last_height - mRPC->first_height)/2);

	aPGeoC2 = cSysCoord::WGS84()->ToGeoC(aPWGS84);
	    
	//collect for intersection
	aVS.push_back(ElSeg3D(aPGeoC1,aPGeoC2));
	aVPds.push_back(1);
    }

    bool aIsOK;
    Pt3dr aRes = ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK);

    if(aIsOK==false)
        std::cout << "not intersected in CameraRPC::OpticalCenterGrid" << std::endl;
    
    return aRes;
}

void CameraRPC::TestDirectRPCGen()
{
    mRPC->TestDirectRPCGen(mCS);
}

bool CameraRPC::HasOpticalCenterOfPixel() const
{
    if(mOptCentersIsDef)
        return true;
    else
	return false;
}

void CameraRPC::FindUTMCS()
{
    //there are some places for which the formula does not work
    ostringstream zone;
    zone << std::floor((mRPC->first_lon + 180)/6 + 1);
	

    if( mRPC->first_lat>0 )
        mCS = "+proj=utm +zone=" + zone.str() + " +ellps=WGS84 +datum=WGS84 +units=m +no_defs";
    else
        mCS = "+proj=utm +zone=" + zone.str() + " +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs";

    std::cout << mCS << std::endl;
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

Pt3dr  CameraAffine::RoughCapteur2Terrain   (const Pt2dr & aP) const
{
  return Pt3dr(0,0,0);
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

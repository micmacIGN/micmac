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

#include "cCameraRPC.h"


bool DEBUG_EWELINA=false;

/* Image coordinates order: [Line, Sample] = [row, col] =  [y, x]*/
/******************************************************/
/*                                                    */
/*                   CameraRPC                        */
/*                                                    */
/******************************************************/

/* create a new constructuor for SPICE */

/* Constructor that takes the RPC in Xml_CamGenPolBundle format */
CameraRPC::CameraRPC(const std::string &aNameFile, const double aAltiSol) :
    mProfondeurIsDef(false),
	mAltisSolIsDef(true),
	mAltiSol(aAltiSol),
	mOptCentersIsDef(false),
	mOpticalCenters(new std::vector<Pt3dr>()),
	mGridSz(Pt2di(10,10)),
    mInputName(aNameFile)
{
    mRPC = new cRPC(aNameFile);

    /* Mean Z */
    SetAltiSol( abs(mRPC->GetGrC31() - mRPC->GetGrC32())*0.5 );
}

/* Constructor that takes original RPC fiels as input */
CameraRPC::CameraRPC(const std::string &aNameFile, 
		             const eTypeImporGenBundle &aType,
		             const cSystemeCoord * aChSys,
                     const double aAltiSol) : 
	mProfondeurIsDef(false),
	mAltisSolIsDef(true),
	mAltiSol(aAltiSol),
	mOptCentersIsDef(false),
	mOpticalCenters(new std::vector<Pt3dr>()),
	mGridSz(Pt2di(10,10)),
	mInputName(aNameFile)
{
    mRPC = new cRPC(aNameFile,aType,aChSys);
    
    /* Mean Z */
    SetAltiSol( abs(mRPC->GetGrC31() - mRPC->GetGrC32())*0.5 );
}

cBasicGeomCap3D * CameraRPC::CamRPCOrientGenFromFile(const std::string & aName, const eTypeImporGenBundle aType, const cSystemeCoord * aChSys)
{
    cBasicGeomCap3D * aRes = new CameraRPC(aName, aType, aChSys); 
    
    return aRes;
}


CameraRPC::~CameraRPC()
{
	delete mRPC;
	delete mOpticalCenters;
}


const cRPC * CameraRPC::GetRPC() const
{
    return mRPC;
}

cRPC CameraRPC::GetRPCCpy() const
{
    return (*mRPC);
}

int CameraRPC::CropRPC(const std::string &aNameDir, 
                        const std::string &aNameXML, 
                        const std::vector<Pt3dr> &aBounds)
{
    int aK;

    /* Some metadata from the Xml */
    cXml_CamGenPolBundle aXml = StdGetFromSI(aNameXML,Xml_CamGenPolBundle);
    const std::string aNameRPC = "Crop-" + NameWithoutDir(aXml.NameCamSsCor());
    const std::string aNameIma = "Crop-" + NameWithoutDir(aXml.NameIma());
    int aDeg = aXml.DegreTot();

    std::string aTmpChSys = "TMPChSys.xml";
    cSystemeCoord *aChSys = aXml.SysCible().PtrVal();
    MakeFileXML(*aChSys,aTmpChSys);

    /* Get 3D grid extent */ 
    Pt3dr aExtMin, aExtMax, aNeverMind;
    cRPC::GetGridExt(aBounds, aExtMin, aExtMax, aNeverMind);

    
    /* Set your grid for RPC size for recalculation */
    Pt3di aRecGrid;
    const Pt3dr aPMin(aExtMin.x, aExtMin.y, mRPC->GetGrC31());
    const Pt3dr aPMax(aExtMax.x, aExtMax.y, mRPC->GetGrC32());
    
    cRPC::SetRecGrid_(mRPC->ISMETER, aPMin, aPMax, aRecGrid);

    /* Get the first "rough" 3D grid
     * backproject the grid to get the min and max in image space 
     * (this is to recalculate RPC starting from a regular grid in image space -> importnt for numerical stability )*/
    std::vector<Pt3dr> aGrid3DRough;
    cRPC::GenGridAbs_(aPMin, aPMax, aRecGrid, aGrid3DRough);
    
    Pt2dr aP;
    Pt3dr aPI1, aPI2;
    std::vector<Pt3dr> aGrid2DRough;
    for(aK=0; aK<int(aGrid3DRough.size()); aK++)
    {
        aP = Ter2Capteur(aGrid3DRough.at(aK));
        if( aP.x >=0 && aP.y >=0 &&
            aP.x < mRPC->mImCols[1] && aP.y < mRPC->mImRows[1] )
            aGrid2DRough.push_back(Pt3dr(aP.x, aP.y, aGrid3DRough.at(aK).z));
    }
    if( aGrid2DRough.size() > 0 )
        cRPC::GetGridExt(aGrid2DRough, aPI1, aPI2, aNeverMind);
    else
        return 0;
    

    /* Create your 2D grid */
    std::vector<Pt3dr> aGrid2D, aGrid2D_;
    cRPC::GenGridAbs_(Pt3dr(int(aPI1.x),int(aPI1.y),mRPC->GetGrC31()), 
                      Pt3dr(int(aPI2.x),int(aPI2.y),mRPC->GetGrC32()), 
                      aRecGrid, aGrid2D);

    /* Create your 2D grid for precision testing */
    std::vector<Pt3dr> aGrid2DTest, aGrid2DTest_;
    Pt3dr aPI1Test( aPI1.x + double(0.5*(aPI2.x-aPI1.x))/aRecGrid.x, 
                    aPI1.y + double(0.5*(aPI2.y-aPI1.y))/aRecGrid.y, 
                    mRPC->GetGrC31() + double(0.5*(mRPC->GetGrC32()-mRPC->GetGrC31()))/aRecGrid.z );
    Pt3dr aPI2Test( aPI2.x - double(0.5*(aPI2.x-aPI1.x))/aRecGrid.x,
                    aPI2.y - double(0.5*(aPI2.y-aPI1.y))/aRecGrid.y,
                    mRPC->GetGrC32() - double(0.5*(mRPC->GetGrC32()-mRPC->GetGrC31()))/aRecGrid.z );
    cRPC::GenGridAbs_(aPI1Test, aPI2Test, Pt3di(aRecGrid.x-1,aRecGrid.y-1,aRecGrid.z-1), aGrid2DTest);

    


    /* Create your 3D grid */
    std::vector<Pt3dr> aGrid3D;
    for(aK=0; aK<int(aGrid2D.size()); aK++)
        aGrid3D.push_back(ImEtZ2Terrain(Pt2dr(aGrid2D.at(aK).x,aGrid2D.at(aK).y),aGrid2D.at(aK).z));

    /* Create your 3D grid for precision testing */
    std::vector<Pt3dr> aGrid3DTest;
    for(aK=0; aK<int(aGrid2DTest.size()); aK++)
        aGrid3DTest.push_back(ImEtZ2Terrain(Pt2dr(aGrid2DTest.at(aK).x, aGrid2DTest.at(aK).y), aGrid2DTest.at(aK).z));

    /* Shift your 2D grid to the origin of the img */
    for(aK=0; aK<int(aGrid2D.size()); aK++)
        aGrid2D_.push_back(Pt3dr(aGrid2D.at(aK).x - int(aPI1.x),
                                 aGrid2D.at(aK).y - int(aPI1.y),
                                 aGrid2D.at(aK).z));

    /* Shift the 2D test grid too */
    for(aK=0; aK<int(aGrid2DTest.size()); aK++)
        aGrid2DTest_.push_back(Pt3dr(aGrid2DTest.at(aK).x - int(aPI1.x),
                                    aGrid2DTest.at(aK).y - int(aPI1.y),
                                    aGrid2DTest.at(aK).z));


    /* Calculate RPCs */
    mRPC->CalculRPC(aGrid3D, aGrid2D_,
                    aGrid3DTest, aGrid2DTest_,
                    mRPC->mDirSNum, mRPC->mDirLNum, mRPC->mDirSDen, mRPC->mDirLDen,
                    mRPC->mInvSNum, mRPC->mInvLNum, mRPC->mInvSDen, mRPC->mInvLDen,
                    1);
    
    /* Save the RPCs */
    cRPC::Save2XmlStdMMName_(*mRPC, aNameRPC);
   

    /* Save the cropped image*/
    Pt3dr aCI1, aCI2;
    cRPC::GetGridExt(aGrid2D, aCI1, aCI2, aNeverMind);
    Pt2di aSzI(aCI2.x - aCI1.x, aCI2.y - aCI1.y);
    Tiff_Im aTiffIm(Tiff_Im::StdConvGen(aXml.NameIma().c_str(),1,false));

    Tiff_Im aTiffCrop
        (
            aNameIma.c_str(),
            aSzI,
            aTiffIm.type_el(),
            Tiff_Im::No_Compr,
            aTiffIm.phot_interp()
         );

    ELISE_COPY
        (
            aTiffCrop.all_pts(),
            trans(aTiffIm.in(),Pt2di(aCI1.x,aCI1.y)),
            aTiffCrop.out()
         );


    /* Run ConvertBundleGen */
    std::string aCom1;
    aCom1 = MM3dBinFile_quotes("Convert2GenBundle") + " " + 
                            aNameIma + " " + aNameRPC + " " + 
                            aNameDir + " ChSys=" + aTmpChSys + 
                            " Degre=" + ToString(aDeg);   
    
    std::cout << "aCom1 " << aCom1 << "\n";

    TopSystem(aCom1.c_str());
    ELISE_fp::RmFile(aTmpChSys); 
   
    return 0;
}

Pt2dr CameraRPC::Ter2Capteur(const Pt3dr & aP) const
{
    AssertRPCInvInit();

    return (mRPC->InverseRPC(aP));
}

bool CameraRPC::PIsVisibleInImage   (const Pt3dr & aP,const cArgOptionalPIsVisibleInImage *) const
{
    // (1) Check if aP is within the RPC validity zone
    /*if( (aP.x < mRPC->first_lon) || (aP.x > mRPC->last_lon) || 
        (aP.y < mRPC->first_lat) || (aP.y > mRPC->last_lat) || 
	(aP.z < mRPC->first_height) || (aP.z > mRPC->last_height) )
        return false;
*/
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

    //beginning of the height validity zone
    double aZ0 = mRPC->GetGrC31()+1;
    if(AltisSolIsDef())
        aZ0 = mAltiSol;
    
    //middle of the height validity zones
    double aZ1 = aZ0+double(mRPC->GetGrC32() - mRPC->GetGrC31())/2;

    return F2toRayonLPH(aP, aZ0, aZ1);
}

ElSeg3D CameraRPC::F2toRayonLPH(const Pt2dr &aP,const double &aZ0, const double &aZ1) const
{
    return(ElSeg3D(mRPC->DirectRPC(aP,aZ0), 
	               mRPC->DirectRPC(aP,aZ1)));
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
    return Pt3dr(0,0,0);
}

Pt3dr CameraRPC::ImEtProf2Terrain(const Pt2dr & aP,double aProf) const
{
    AssertRPCDirInit();

    if(mOptCentersIsDef)
    {
        //find the  sensor's mean Z-position
	    unsigned int aK;
	    double aZ(mOpticalCenters->at(0).z);
	    for(aK=1; aK<mOpticalCenters->size(); aK++)
	        aZ += mOpticalCenters->at(aK).z;

	    const double aZc = double(aZ)/mOpticalCenters->size() - aProf;

        return(mRPC->DirectRPC(aP, aZc));

    }
    else
    {
// MPD pour corriger bug reproj epipo dans Saisie Appuis 
// A terme il faudra changer avec une fonction specifique
        Pt3dr aC  = OpticalCenterOfPixel(aP);
        ElSeg3D   aSeg = Capteur2RayTer(aP);
        Pt3dr aPTer = RoughCapteur2Terrain(aP);

        double aProfTer = euclid(aPTer-aC);
        double aProp = aProf / ProfondeurDeChamps(aPTer);
        double aDist = aProfTer * (1 - aProp);
        
        return aPTer  + aSeg.TgNormee() * aDist;
        //ElSeg3D aSeg(aC


        ELISE_ASSERT(false,"CameraRPC::ImEtProf2Terrain no data about the sensor positon");
	
	return(Pt3dr(0,0,0));
    }
}

Pt3dr CameraRPC::ImEtZ2Terrain(const Pt2dr & aP, double aZ) const
{
    AssertRPCDirInit();

    return(mRPC->DirectRPC(aP, aZ));
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
	int aK;
    
	mAltiSol = aZ;
    mAltisSolIsDef = true;
	


	Box2dr aBox(Pt2dr(0,0),Pt2dr(SzBasicCapt3D()));
	Pt2dr aP4Im[4];
	aBox.Corners(aP4Im);


	if (mContourUtile.empty())
	{
		for (aK=0 ; aK<4 ; aK++)
			mContourUtile.push_back(aP4Im[aK]);
	}



	Pt2dr aP0,aP1;
	std::vector<Pt2dr>  aCont;	

	for (aK=0 ; aK<int(ContourUtile().size()) ; aK++)
	{
		Pt2dr aCk= ContourUtile()[aK];

		Pt3dr aPTer = ImEtZ2Terrain(aCk,aZ);
		Pt2dr aP2T(aPTer.x,aPTer.y);
		if (aK==0)
		{
			aP0 = aP2T;
			aP1 = aP2T;
		}
		else
		{
			aP0.SetInf(aP2T);
			aP1.SetSup(aP2T);
		}
		aCont.push_back(aP2T);
	}
	mBoxSol = Box2dr(aP0,aP1);
	mEmpriseSol = cElPolygone();
	mEmpriseSol.AddContour(aCont,false);
}

const cElPolygone &  CameraRPC::EmpriseSol() const
{
	return mEmpriseSol;
}

const Box2dr &  CameraRPC::BoxSol() const
{
	return mBoxSol;
}

const std::vector<Pt2dr> &  CameraRPC::ContourUtile()
{
	ELISE_ASSERT(!mContourUtile.empty(),"CameraRPC::ContourUtile non init");

	return mContourUtile;

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
    if( (aP.y >= mRPC->GetImRow1()) && (aP.y <= mRPC->GetImRow2()) &&
        (aP.x >= mRPC->GetImCol1()) && (aP.x <= mRPC->GetImCol2()))
        return true;
    else
	    return false;
}

void CameraRPC::AssertRPCDirInit() const
{
    ELISE_ASSERT(mRPC->IsDir(),"CameraRPC::AssertRPCDirInit");
}

void CameraRPC::AssertRPCInvInit() const
{
    ELISE_ASSERT(mRPC->IsInv(),"CameraRPC::AssertRPCInvInit");
}

Pt2di CameraRPC::SzBasicCapt3D() const
{
    ELISE_ASSERT(mRPC!=0,"RPCs were not initialized in CameraRPC::SzBasicCapt3D()");

    return  (Pt2di(mRPC->GetImCol2(),
 	               mRPC->GetImRow2()));
}

 
void CameraRPC::OpticalCenterOfImg()
{
    int aL, aS, aAd = 1;
    double aSStep = double(SzBasicCapt3D().x)/(mGridSz.x+aAd);
    Pt3dr aPWGS84, aPGeoC1, aPGeoC2;
    
    for( aL=0; aL<SzBasicCapt3D().y; aL++)
    {
        std::vector<double> aVPds;
	std::vector<ElSeg3D> aVS;

	for( aS=aAd; aS<mGridSz.x+aAd; aS++)
	{
    
	
	    //first height in validity zone
	    aPWGS84 = ImEtZ2Terrain(Pt2dr(aS*aSStep,aL),
			                  mRPC->GetGrC31()+1);
	    aPGeoC1 = cSysCoord::WGS84Degre()->ToGeoC(aPWGS84);
            
	    //second height in validity zone
	    aPWGS84 = ImEtZ2Terrain(Pt2dr(aS*aSStep,aL), 
			           (mRPC->GetGrC31()+1) +
				   double(mRPC->GetGrC32() - 
					  mRPC->GetGrC31())/2);

	    aPGeoC2 = cSysCoord::WGS84Degre()->ToGeoC(aPWGS84);
	    
	    //collect for intersection
	    aVS.push_back(ElSeg3D(aPGeoC1,aPGeoC2));
	    aVPds.push_back(1);
	}

        
	bool aIsOK;
	mOpticalCenters->push_back( ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK) );

	if(aIsOK==false)
	    std::cout << "not intersected in CameraRPC::OpticalCenterOfImg()" << std::endl;

    }

    mOptCentersIsDef=true; 
}


/* For a defined image grid, 
 * extrude to rays and intersect at the line of optical centers */
void CameraRPC::OpticalCenterGrid(const Pt2di& aGrid, bool aIfSave) const
{
    srand(time(NULL));
    int aL, aS;
    int aCRand1 = rand() % 255, aCRand2 = rand() % 255, aCRand3 = rand() % 255;
 
    std::vector<Pt3dr> aOptCentersAtGrid;
    std::vector<Pt3di> aVCol;
    Pt3di aCoul(aCRand1,aCRand2,aCRand3);
    std::vector<const cElNuage3DMaille *> aVNuage;

    std::string aDir = "PBoptCenter//";
    std::string aPlyFile = aDir + "OptCenter_" + mInputName + ".ply";
    std::list<std::string> aVCom;

    ELISE_fp::MkDirSvp(aDir);
    
    int aAd=1;
    Pt2dr aGridStep = Pt2dr( double(SzBasicCapt3D().x)/(aGrid.x+aAd) ,
                             double(SzBasicCapt3D().y)/(aGrid.y+aAd));

    Pt3dr aP1, aP2;
    for( aL=aAd; aL<aGrid.y+aAd; aL++)
    {
        std::vector<double> aVPds;
        std::vector<ElSeg3D> aVS;
	    
	for( aS=aAd; aS<aGrid.x+aAd; aS++)
        {
            //first height in validity zone
	        aP1 = ImEtZ2Terrain(Pt2dr(aS*aGridStep.x,aL*aGridStep.y),
			                  mRPC->GetGrC31()+1);
	    
            
	        //second height in validity zone
	        aP2 = ImEtZ2Terrain(Pt2dr(aS*aGridStep.x,aL*aGridStep.y), 
			           (mRPC->GetGrC31()+1)+double(mRPC->GetGrC32() - mRPC->GetGrC31())/2);

	        //collect for intersection
	        aVS.push_back(ElSeg3D(aP1,aP2));
	        aVPds.push_back(1);

            aVCol.push_back(aCoul);
	        aVCol.push_back(aCoul);

        }
        
	bool aIsOK;
	aOptCentersAtGrid.push_back( ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK) );

	if(aIsOK==false)
	    std::cout << "not intersected in CameraRPC::OpticalCenterGrid" << std::endl;
    

     }

    //save to ply
    cElNuage3DMaille::PlyPutFile
    (
       aPlyFile,
       aVCom,
       aVNuage,
       &aOptCentersAtGrid,
       &aVCol,
       true
    );


}


Pt3dr CameraRPC::OpticalCenterOfPixel(const Pt2dr & aP) const
{
    if(mOptCentersIsDef==true)
    {
	    int aLnPre, aLnPost;
	    double aA, aB, aC, aABC, aT; 

	    (aP.y > 1) ? (aLnPre = round_down(aP.y) - 1) : aLnPre = 0;
	
	    (aP.y < (SzBasicCapt3D().y-2)) ? (aLnPost = round_up(aP.y) + 1) : 
		                         (aLnPost = SzBasicCapt3D().y-1);
        
        aA = mOpticalCenters->at(aLnPre).x - mOpticalCenters->at(aLnPost).x;
        aB = mOpticalCenters->at(aLnPre).y - mOpticalCenters->at(aLnPost).y;
        aC = mOpticalCenters->at(aLnPre).z - mOpticalCenters->at(aLnPost).z;
	
	    aABC = std::sqrt(aA*aA + aB*aB + aC*aC);

        aA /= aABC;
        aB /= aABC;
        aC /= aABC;

	    aT = double(aP.y - aLnPre)/(aLnPost - aLnPre);

	    
        return Pt3dr(mOpticalCenters->at(aLnPre).x + aT*aA,
		            mOpticalCenters->at(aLnPre).y + aT*aB,
		            mOpticalCenters->at(aLnPre).z + aT*aC);
    }
    else
    { 
        int aS, aAd = 1;
        double aSStep = double(SzBasicCapt3D().x)/(mGridSz.x+aAd);
        Pt3dr aP1, aP2;

        std::vector<double> aVPds;
        std::vector<ElSeg3D> aVS;

        for(aS=aAd; aS<mGridSz.x+aAd; aS++)
        {
    
            //first height in validity zone
    	    aP1 = ImEtZ2Terrain(Pt2dr(aS*aSStep,aP.y),
	 		            mRPC->GetGrC31()+1);
	    
            
	        //second height in validity zone
	        aP2 = ImEtZ2Terrain(Pt2dr(aS*aSStep,aP.y), 
			           (mRPC->GetGrC31()+1) + 
				   double(mRPC->GetGrC32() - mRPC->GetGrC31())/2);

	    
	        //collect for intersection
	        aVS.push_back(ElSeg3D(aP1,aP2));
	        aVPds.push_back(1);
        }

        bool aIsOK;
        Pt3dr aRes = ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK);

        if(aIsOK==false)
            std::cout << "not intersected in CameraRPC::OpticalCenterOfPixel" << std::endl;

//	std::cout << aRes << " " << "\n";	
        return aRes;
    }
}

/*void CameraRPC::TestDirectRPCGen()
{
    mRPC->TestDirectRPCGen();
}*/

bool CameraRPC::HasOpticalCenterOfPixel() const
{
    if(mOptCentersIsDef)
        return true;
    else
	    return false;
}


void CameraRPC::ExpImp2Bundle(std::vector<std::vector<ElSeg3D> > aGridToExp) const
{
    AssertRPCDirInit();
    ELISE_ASSERT(mRPC->IsMetric()," CameraRPC::ExpImp2Bundle, the RPC and the grid must be in meters");


    Pt3di aGridSz = mRPC->GetGrid();
    Pt2dr aGridStep = Pt2dr( double(SzBasicCapt3D().x)/aGridSz.x,
                             double(SzBasicCapt3D().y)/aGridSz.y );

    int aK1=0, aK2=0;

    if( aGridToExp.size()==0 )
    {
        aGridToExp.resize(aGridSz.y);
        for(aK1=0; aK1<aGridSz.y; aK1++)
        {
            for(aK2=0; aK2<aGridSz.x; aK2++)
            {
                aGridToExp.at(aK1).push_back( Capteur2RayTer( Pt2dr(aK2*aGridStep.x,aK1*aGridStep.y) ));
            }
        
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
        aSLS.GridSz() = Pt2di(aGridSz.x,aGridSz.y);

        for( aK1=0; aK1<aGridSz.y; aK1++ )
        {
            cXml_OneLineSLS aOL;
            aOL.IndLine() = aK1*aGridStep.y;
            for( aK2=0; aK2<aGridSz.x; aK2++ )
            {
                cXml_SLSRay aOR;
                aOR.IndCol() = aK2*aGridStep.x;

                aOR.P1() = aGridToExp.at(aK1).at(aK2).P0();
                aOR.P2() = aGridToExp.at(aK1).at(aK2).P1();

                aOL.Rays().push_back(aOR);
            }
            aSLS.Lines().push_back(aOL);
        }
        //export to XML format
        std::string aXMLSave = DirOfFile(mInputName) + "Bundles-" + StdPrefix(NameWithoutDir(mInputName)) + ".xml";
        MakeFileXML(aSLS, aXMLSave);
    }
}

void CameraRPC::Save2XmlStdMMName(const std::string &aName,const std::string & aPref) const
{
    mRPC->Save2XmlStdMMName(aName,aPref);
}

/******************************************************/
/*                                                    */
/*                  CameraAffine                      */
/*                                                    */
/******************************************************/
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

bool CameraAffine::PIsVisibleInImage   (const Pt3dr & aP,const cArgOptionalPIsVisibleInImage *) const
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

Pt3dr CameraAffine::RoughCapteur2Terrain(const Pt2dr & aP) const
{
    return Pt3dr(0,0,0);
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


/******************************************************/
/*                                                    */
/*                    cRPC                            */
/*                                                    */
/******************************************************/

/* Direct algoritm
 * ground_coord_1 = mDirSNum / mDirSDen
 * ground_coord_2 = mDirLNum / mDirLDen
 *
 * inverse algorithm
 * image_col = mInvSNum / mInvSDen
 * image_row = mInvLNum / mInvLDen  */

/* Constructor that takes cXml_CamGenPolBundle or OrientationGrille 
 * as input */
cRPC::cRPC(const std::string &aName) :
    ISDIR(false),
    ISINV(false),
    ISMETER(false),
    mRefine(eRP_None),
    mRecGrid(Pt3di(0,0,0)),
    mName("")
{
    if( AutoDetermineRPCFile(aName) )
    {
        /* Read Xml_CamGenPolBundle */
        cXml_CamGenPolBundle aXml = StdGetFromSI(aName,Xml_CamGenPolBundle);
        const std::string aNameRPC = aXml.NameCamSsCor();
        mName = aNameRPC;
        ELISE_ASSERT(ELISE_fp::exist_file(aNameRPC),"cRPC::cRPC; the file with the polyn does not exist!");

        eTypeImporGenBundle aType;
        bool aModeHelp;
       
        /* Retrieve the coordinate system for the computations */
        const cSystemeCoord * aChSys = aXml.SysCible().PtrCopy();//Val();
        

        if(AutoDetermineGRIDFile(aNameRPC))
        {
            std::string aNameType="TIGB_MMOriGrille";
            StdReadEnum(aModeHelp,aType,aNameType,eTIGB_NbVals);
        }
        else
        {
            /* Retrieve RPC format */
            StdReadEnum(aModeHelp,aType,"TIGB_Unknown",eTIGB_NbVals);
            AutoDetermineTypeTIGB(aType,aXml.NameCamSsCor());
        }

        /* Initialize the class variables */
        Initialize(aName,aType,aChSys);

    }
    else
        ELISE_ASSERT( false, "cRPC::cRPC; this constructor needs RPC in cXml_CamGenPolBundle format or the grid file" );
    
    
}


/* Constructor to maintain Convert2GenBundle */
cRPC::cRPC(const std::string &aName, const eTypeImporGenBundle &aType, const cSystemeCoord *aChSys) :
    ISDIR(false),
    ISINV(false),
    ISMETER(false),
    mRefine(eRP_None),
    mRecGrid(Pt3di(0,0,0)),
    mName(aName)
{
    /* Initialize the class variables */
    Initialize(aName,aType,aChSys);


}

void cRPC::Initialize(const std::string &aName, 
                      const eTypeImporGenBundle &aType,
                      const cSystemeCoord *aChSys
                      )
{
    std::string aNameRPC=aName;
    if(AutoDetermineRPCFile(aName))
    {
        cXml_CamGenPolBundle aXml = StdGetFromSI(aName,Xml_CamGenPolBundle);
        aNameRPC=aXml.NameCamSsCor();
        
        mPol = cPolynomial_BGC3M2D::NewFromFile(aName);
        
        mRefine = eRP_Poly;

    }
    
    if(aChSys)
        mChSys = *aChSys;

    if(aType==eTIGB_MMDimap2)
    {
        ReadDimap(aNameRPC);
	//Show();       
        Initialize_(aChSys);

    }
    else if(aType==eTIGB_MMDimap1)
    {
        ELISE_ASSERT(false,"CameraRPC::Initialize; No eTIGB_MMDimap1");
    }
    else if(aType==eTIGB_MMDGlobe)
    {
        ReadXML(aNameRPC);
    	
	    SetRecGrid();
	    
        InvToDirRPC();
        //Show(); 
        Initialize_(aChSys);

    }
    else if(aType==eTIGB_MMIkonos)
    {
        ReadASCII(aNameRPC);
	    
        //ReadASCIIMeta("Metafile.txt", aNameRPC);

	    SetRecGrid();
    
        InvToDirRPC();
	//Show();
        Initialize_(aChSys);

    }
    else if(aType==eTIGB_MMEuclid)
    {
        ReadEUCLIDIUM(aNameRPC);
        //Show(); 
        Initialize_(aChSys);
    
    }
    else if(aType==eTIGB_MMOriGrille)
    {
        ISMETER = true;

        OrientationGrille aGrill(aNameRPC);
      
        /* Set validities */
        mGrC1[0] = aGrill.GetRangeX().x;
        mGrC1[1] = aGrill.GetRangeX().y;
        mGrC2[0] = aGrill.GetRangeY().x;
        mGrC2[1] = aGrill.GetRangeY().y;
        mGrC3[0] = aGrill.GetRangeZ().x;
        mGrC3[1] = aGrill.GetRangeZ().y;

        mImRows[0] = aGrill.GetRangeRow().x;
        mImRows[1] = aGrill.GetRangeRow().y;
        mImCols[0] = aGrill.GetRangeCol().x;
        mImCols[1] = aGrill.GetRangeCol().y;

        /* Set the reconstruction grid size*/
        SetRecGrid();
        
        /* Grid in 3D */
        std::vector<Pt3dr> aGrid3D;
        GenGridAbs(mRecGrid, aGrid3D);
       
        /* Grid in 2D */
        int aK;
        Pt2dr aP;
        std::vector<Pt3dr> aGrid2D;
        for(aK=0; aK< int(aGrid3D.size()); aK++)
        {
            aGrill.Objet2ImageInit(aGrid3D.at(aK).x, aGrid3D.at(aK).y, &(aGrid3D.at(aK).z),
                            aP.x, aP.y);
            aGrid2D.push_back(Pt3dr(aP.x, aP.y, aGrid3D.at(aK).z));
        }
        

        /* Learn parameters */
        std::vector<Pt3dr> aGrid2DTest, aGrid3DTest;//vectors empty so no test will be done
        CalculRPC(aGrid3D, aGrid2D,
                  aGrid3DTest, aGrid2DTest,
                  mDirSNum, mDirLNum, mDirSDen, mDirLDen,
                  mInvSNum, mInvLNum, mInvSDen, mInvLDen, 1);

        Show();

        ISDIR=true;
        ISINV=true;

    }
    else if(aType==eTIGB_MMSpice)
    {
    
    }
    /*else if (aType == eTIGB_MMASTER)
    {
	   AsterMetaDataXML(aNameFile);
	   std::string aNameIm = StdPrefix(aNameFile) + ".tif";
	   //Find Validity and normalization values
	   ComputeNormFactors(0, 8000);//0 and 800 are min and max altitude
	   vector<vector<Pt3dr> > aGridNorm = 
		                  mRPC->GenerateNormLineOfSightGrid(20,0,8000);
	   GCP2Direct(aGridNorm[0], aGridNorm[1]);
	   GCP2Inverse(aGridNorm[0], aGridNorm[1]);
    }*/
    else {ELISE_ASSERT(false,"Unknown RPC mode.");}

    //Show();
    
}

void cRPC::Initialize_(const cSystemeCoord *aChSys)
{
    SetRecGrid();

    if(aChSys)
        ChSysRPC(mChSys);
}

std::string cRPC::NameSave(const std::string & aName)
{
    std::string aNewDir = DirOfFile(aName)+ "NEW/";
    ELISE_fp::MkDirSvp(aNewDir);

    std::string aNameXml = aNewDir + StdPrefix(NameWithoutDir(aName)) + ".xml";
    
    return aNameXml;
}

template <typename T>
void cRPC::FilLineNumCoeff(T& aXml, double (&aLNC)[20]) const
{
   aXml.LINE_NUM_COEFF_1() = aLNC[0];  
   aXml.LINE_NUM_COEFF_2() = aLNC[1];  
   aXml.LINE_NUM_COEFF_3() = aLNC[2]; 
   aXml.LINE_NUM_COEFF_4() = aLNC[3]; 
   aXml.LINE_NUM_COEFF_5() = aLNC[4]; 
   aXml.LINE_NUM_COEFF_6() = aLNC[5]; 
   aXml.LINE_NUM_COEFF_7() = aLNC[6]; 
   aXml.LINE_NUM_COEFF_8() = aLNC[7]; 
   aXml.LINE_NUM_COEFF_9() = aLNC[8];  
   aXml.LINE_NUM_COEFF_10() = aLNC[9];  
   aXml.LINE_NUM_COEFF_11() = aLNC[10];  
   aXml.LINE_NUM_COEFF_12() = aLNC[11];  
   aXml.LINE_NUM_COEFF_13() = aLNC[12];  
   aXml.LINE_NUM_COEFF_14() = aLNC[13];  
   aXml.LINE_NUM_COEFF_15() = aLNC[14];  
   aXml.LINE_NUM_COEFF_16() = aLNC[15];  
   aXml.LINE_NUM_COEFF_17() = aLNC[16];  
   aXml.LINE_NUM_COEFF_18() = aLNC[17];  
   aXml.LINE_NUM_COEFF_19() = aLNC[18];
   aXml.LINE_NUM_COEFF_20() = aLNC[19];
}

template <typename T>
void cRPC::FilLineDenCoeff(T& aXml, double (&aLDC)[20]) const
{
   aXml.LINE_DEN_COEFF_1() = aLDC[0];  
   aXml.LINE_DEN_COEFF_2() = aLDC[1];  
   aXml.LINE_DEN_COEFF_3() = aLDC[2];  
   aXml.LINE_DEN_COEFF_4() = aLDC[3];  
   aXml.LINE_DEN_COEFF_5() = aLDC[4];  
   aXml.LINE_DEN_COEFF_6() = aLDC[5];  
   aXml.LINE_DEN_COEFF_7() = aLDC[6];  
   aXml.LINE_DEN_COEFF_8() = aLDC[7];  
   aXml.LINE_DEN_COEFF_9() = aLDC[8];  
   aXml.LINE_DEN_COEFF_10() = aLDC[9];  
   aXml.LINE_DEN_COEFF_11() = aLDC[10];  
   aXml.LINE_DEN_COEFF_12() = aLDC[11];  
   aXml.LINE_DEN_COEFF_13() = aLDC[12];  
   aXml.LINE_DEN_COEFF_14() = aLDC[13];  
   aXml.LINE_DEN_COEFF_15() = aLDC[14];  
   aXml.LINE_DEN_COEFF_16() = aLDC[15];  
   aXml.LINE_DEN_COEFF_17() = aLDC[16];  
   aXml.LINE_DEN_COEFF_18() = aLDC[17];  
   aXml.LINE_DEN_COEFF_19() = aLDC[18];  
   aXml.LINE_DEN_COEFF_20() = aLDC[19];  
}

template <typename T>
void cRPC::FilSampNumCoeff(T& aXml, double (&aSNC)[20]) const
{

   aXml.SAMP_NUM_COEFF_1() = aSNC[0];  
   aXml.SAMP_NUM_COEFF_2() = aSNC[1];  
   aXml.SAMP_NUM_COEFF_3() = aSNC[2];  
   aXml.SAMP_NUM_COEFF_4() = aSNC[3];  
   aXml.SAMP_NUM_COEFF_5() = aSNC[4];  
   aXml.SAMP_NUM_COEFF_6() = aSNC[5];  
   aXml.SAMP_NUM_COEFF_7() = aSNC[6];  
   aXml.SAMP_NUM_COEFF_8() = aSNC[7];  
   aXml.SAMP_NUM_COEFF_9() = aSNC[8];  
   aXml.SAMP_NUM_COEFF_10() = aSNC[9];  
   aXml.SAMP_NUM_COEFF_11() = aSNC[10];  
   aXml.SAMP_NUM_COEFF_12() = aSNC[11];  
   aXml.SAMP_NUM_COEFF_13() = aSNC[12];  
   aXml.SAMP_NUM_COEFF_14() = aSNC[13];  
   aXml.SAMP_NUM_COEFF_15() = aSNC[14];  
   aXml.SAMP_NUM_COEFF_16() = aSNC[15];  
   aXml.SAMP_NUM_COEFF_17() = aSNC[16];  
   aXml.SAMP_NUM_COEFF_18() = aSNC[17];  
   aXml.SAMP_NUM_COEFF_19() = aSNC[18];  
   aXml.SAMP_NUM_COEFF_20() = aSNC[19];  
}

template <typename T>
void cRPC::FilSampDenCoeff(T& aXml, double (&aSDC)[20]) const
{

   aXml.SAMP_DEN_COEFF_1() = aSDC[0];  
   aXml.SAMP_DEN_COEFF_2() = aSDC[1];  
   aXml.SAMP_DEN_COEFF_3() = aSDC[2];  
   aXml.SAMP_DEN_COEFF_4() = aSDC[3];  
   aXml.SAMP_DEN_COEFF_5() = aSDC[4];  
   aXml.SAMP_DEN_COEFF_6() = aSDC[5];  
   aXml.SAMP_DEN_COEFF_7() = aSDC[6];  
   aXml.SAMP_DEN_COEFF_8() = aSDC[7];  
   aXml.SAMP_DEN_COEFF_9() = aSDC[8];  
   aXml.SAMP_DEN_COEFF_10() = aSDC[9];  
   aXml.SAMP_DEN_COEFF_11() = aSDC[10];  
   aXml.SAMP_DEN_COEFF_12() = aSDC[11];  
   aXml.SAMP_DEN_COEFF_13() = aSDC[12];  
   aXml.SAMP_DEN_COEFF_14() = aSDC[13];  
   aXml.SAMP_DEN_COEFF_15() = aSDC[14];  
   aXml.SAMP_DEN_COEFF_16() = aSDC[15];  
   aXml.SAMP_DEN_COEFF_17() = aSDC[16];  
   aXml.SAMP_DEN_COEFF_18() = aSDC[17];  
   aXml.SAMP_DEN_COEFF_19() = aSDC[18];  
   aXml.SAMP_DEN_COEFF_20() = aSDC[19];  
}

void cRPC::Save2XmlStdMMName_(cRPC &aRPC, const std::string &aName)
{
   /* Change the coordinate sytem to original geodetic */
   aRPC.ChSysRPC(aRPC.mChSys);
   
   cXml_RPC aXml_RPC;
   cXml_RPC_Coeff aXml_Dir, aXml_Inv;
   cXml_RPC_Validity aXml_Val;

   aXml_RPC.METADATA_FORMAT() = "DIMAP";
   aXml_RPC.METADATA_VERSION() = "2.0";

   /* Direct */
   aRPC.FilSampNumCoeff(aXml_Dir.SAMP_NUM_COEFF(),(aRPC.mDirSNum));
   
   aRPC.FilSampDenCoeff(aXml_Dir.SAMP_DEN_COEFF(),aRPC.mDirSDen);

   aRPC.FilLineNumCoeff(aXml_Dir.LINE_NUM_COEFF(),aRPC.mDirLNum);
   
   aRPC.FilLineDenCoeff(aXml_Dir.LINE_DEN_COEFF(),aRPC.mDirLDen);

   aXml_RPC.Direct_Model() = aXml_Dir;

   /* Inverse */
   aRPC.FilSampNumCoeff(aXml_Inv.SAMP_NUM_COEFF(),aRPC.mInvSNum);
   
   aRPC.FilSampDenCoeff(aXml_Inv.SAMP_DEN_COEFF(),aRPC.mInvSDen);

   aRPC.FilLineNumCoeff(aXml_Inv.LINE_NUM_COEFF(),aRPC.mInvLNum);

   aRPC.FilLineDenCoeff(aXml_Inv.LINE_DEN_COEFF(),aRPC.mInvLDen);

   aXml_RPC.Inverse_Model() = aXml_Inv;


   /* Validity */
   vector<double> aImOff, aImScal, aGrOff, aGrScal;
   vector<double> aImRows, aImCols, aGrC1, aGrC2, aGrC3;
   
   aRPC.GetGrC1(aGrC1);
   aRPC.GetGrC2(aGrC2);
   aGrC3.push_back(aRPC.GetGrC31());
   aGrC3.push_back(aRPC.GetGrC32());
   aRPC.GetImOff(aImOff);
   aRPC.GetImScal(aImScal);
   aRPC.GetGrOff(aGrOff);
   aRPC.GetGrScal(aGrScal);

   /* Direct */
   aXml_Val.FIRST_ROW() = aRPC.GetImRow1();
   aXml_Val.LAST_ROW() = aRPC.GetImRow2();

   aXml_Val.FIRST_COL() = aRPC.GetImCol1();
   aXml_Val.LAST_COL() = aRPC.GetImCol2();

   aXml_Val.SAMP_SCALE() = aImScal.at(0);
   aXml_Val.SAMP_OFF() = aImOff.at(0);
 
   aXml_Val.LINE_SCALE() = aImScal.at(1);
   aXml_Val.LINE_OFF() = aImOff.at(1);
   
   /* Inverse */
   aXml_Val.FIRST_LON() = aGrC1.at(0);
   aXml_Val.LAST_LON() = aGrC1.at(1);

   aXml_Val.FIRST_LAT() = aGrC2.at(0);
   aXml_Val.LAST_LAT() = aGrC2.at(1);

   aXml_Val.LONG_SCALE() = aGrScal.at(0);
   aXml_Val.LONG_OFF() = aGrOff.at(0);

   aXml_Val.LAT_SCALE() = aGrScal.at(1);
   aXml_Val.LAT_OFF() = aGrOff.at(1);
   
   aXml_Val.HEIGHT_SCALE() = aGrScal.at(2);
   aXml_Val.HEIGHT_OFF() = aGrOff.at(2);

   aXml_RPC.RFM_Validity() = aXml_Val;

   MakeFileXML(aXml_RPC,aName);
   std::cout << "Saved to: " << aName << "\n";
    
}

void cRPC::Save2XmlStdMMName(const std::string &aName,const std::string & aPref)
{
    /* Create new RPC */
    cRPC aRPCSauv(aName);
   
    std::string aNameXml = cRPC::NameSave(aRPCSauv.mName);
    std::string aNewDirLoc = DirOfFile(aNameXml); 
  
    /* Save the new RPC to XML file */
    cRPC::Save2XmlStdMMName_(aRPCSauv,aNameXml);

    /* Save the new cXml_CamGenPolBundle file :
     * - read the old cXml_CamGenPolBundle,
     * - copy it (partially) to the new cXml_CamGenPolBundle, 
     * - save the new cXml_CamGenPolBundle */
    cXml_CamGenPolBundle aXML =  StdGetFromSI(aName,Xml_CamGenPolBundle);

    int aType = eTIGB_Unknown;
    cBasicGeomCap3D * aCamSsCor = cBasicGeomCap3D::StdGetFromFile(aNameXml,aType,aXML.SysCible().PtrCopy());
    const cSystemeCoord * aCh = aXML.SysCible().PtrCopy();

    
    cPolynomial_BGC3M2D aPolNew(aCh,aCamSsCor,aNameXml,aXML.NameIma(),0);
    std::string aNameGenXml =  aPolNew.NameSave("","");
    ELISE_fp::RmDir(DirOfFile(aNameGenXml));

    
    cXml_CamGenPolBundle aXMLNew = aPolNew.ToXml();


    MakeFileXML(aXMLNew,aNewDirLoc+NameWithoutDir(aNameGenXml));


}

bool cRPC::AutoDetermineGRIDFile(const std::string &aFile) const
{
    std::string aPost = StdPostfix(aFile.substr(0,aFile.size()-4));
    if( aPost != "txt" && aPost != "TXT" && aPost != "Txt" && aPost != "rpc" )
    { 
        cElXMLTree * aTree = new cElXMLTree(aFile);

        cElXMLTree * aFirstChild = aTree->Get("trans_coord");
    
        if(aFirstChild)
        {
            return true;
        }
        else
            return false;
    }
    else
	return false; 
   /*
    XercesDOMParser* parser = new XercesDOMParser();
    parser->parse(aFile.c_str());
    DOMNode* doc = parser->getDocument();
    DOMNode* n = doc->getFirstChild();

    if (n)
        n=n->getFirstChild();

    if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("trans_coord")))
        return true;
    else
        return false;
*/

}

bool cRPC::AutoDetermineRPCFile(const std::string &aFile) const
{
    if( (aFile.substr(aFile.size()-3,3) == "XML") ||
        (aFile.substr(aFile.size()-3,3) == "xml"))
    {
        if( (aFile.substr(aFile.size()-7,3) != "TXT") &&
            (aFile.substr(aFile.size()-7,3) != "txt") &&
            (aFile.substr(aFile.size()-7,3) != "rpc"))
        {
            cElXMLTree * aTree = new cElXMLTree(aFile);

            cElXMLTree * aXmlMETADATA_FORMAT = aTree->Get("Xml_CamGenPolBundle");
            if (aXmlMETADATA_FORMAT)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
            return false;
    }
    else
        return false;
}

void cRPC::CubicPolyFil(const Pt3dr &aP, double (&aPTerms)[20]) const
{
    aPTerms[0] = 1;
    aPTerms[1] = aP.x;
    aPTerms[2] = aP.y; 
    aPTerms[3] = aP.z;
    aPTerms[4] = aP.y*aP.x;
    aPTerms[5] = aP.x*aP.z;
    aPTerms[6] = aP.y*aP.z;
    aPTerms[7] = aP.x*aP.x;
    aPTerms[8] = aP.y*aP.y;
    aPTerms[9] = aP.z*aP.z;
    aPTerms[10] = aP.x*aP.y*aP.z;
    aPTerms[11] = aP.x*aP.x*aP.x;
    aPTerms[12] = aP.y*aP.y*aP.x;
    aPTerms[13] = aP.x*aP.z*aP.z;
    aPTerms[14] = aP.x*aP.x*aP.y;
    aPTerms[15] = aP.y*aP.y*aP.y;
    aPTerms[16] = aP.y*aP.z*aP.z;
    aPTerms[17] = aP.x*aP.x*aP.z;
    aPTerms[18] = aP.y*aP.y*aP.z;
    aPTerms[19] = aP.z*aP.z*aP.z;
    
}

void cRPC::DifCubicPolyFil(const Pt3dr &aP, double &aB, double (&aPTerms)[39]) const
{
    double aPoly1[20];

    CubicPolyFil(aP, aPoly1);


    aPTerms[0] = aPoly1[0];
    aPTerms[1] = aPoly1[1];
    aPTerms[2] = aPoly1[2];
    aPTerms[3] = aPoly1[3];
    aPTerms[4] = aPoly1[4];
    aPTerms[5] = aPoly1[5];
    aPTerms[6] = aPoly1[6];
    aPTerms[7] = aPoly1[7];
    aPTerms[8] = aPoly1[8];
    aPTerms[9] = aPoly1[9];
    aPTerms[10] = aPoly1[10];
    aPTerms[11] = aPoly1[11];
    aPTerms[12] = aPoly1[12];
    aPTerms[13] = aPoly1[13];
    aPTerms[14] = aPoly1[14];
    aPTerms[15] = aPoly1[15];
    aPTerms[16] = aPoly1[16];
    aPTerms[17] = aPoly1[17];
    aPTerms[18] = aPoly1[18];
    aPTerms[19] = aPoly1[19];

    aPTerms[20] = -aB * aPoly1[1];
    aPTerms[21] = -aB * aPoly1[2];
    aPTerms[22] = -aB * aPoly1[3];
    aPTerms[23] = -aB * aPoly1[4];
    aPTerms[24] = -aB * aPoly1[5];
    aPTerms[25] = -aB * aPoly1[6];
    aPTerms[26] = -aB * aPoly1[7];
    aPTerms[27] = -aB * aPoly1[8];
    aPTerms[28] = -aB * aPoly1[9];
    aPTerms[29] = -aB * aPoly1[10];
    aPTerms[30] = -aB * aPoly1[11];
    aPTerms[31] = -aB * aPoly1[12];
    aPTerms[32] = -aB * aPoly1[13];
    aPTerms[33] = -aB * aPoly1[14];
    aPTerms[34] = -aB * aPoly1[15];
    aPTerms[35] = -aB * aPoly1[16];
    aPTerms[36] = -aB * aPoly1[17];
    aPTerms[37] = -aB * aPoly1[18];
    aPTerms[38] = -aB * aPoly1[19];
                        
}

/* R2 -> R3 (Sample, Line -> Ground_Coords) */
Pt3dr cRPC::DirectRPC(const Pt2dr &aP, const double &aZ) const
{
    Pt2dr aPIm(aP.x, aP.y);
    Pt3dr aPGr, aPGr_;

    //polynom refinement
    if(mRefine==eRP_Poly)
    {
        Pt2dr aDep=mPol->DeltaCamInit2CurIm(Pt2dr(aP.x, aP.y));
        aPIm -= aDep; 
    }
    //normalize
    aPIm = NormIm(aPIm);
    const double aZGr = NormGrZ(aZ);

    //apply direct RPCs
    aPGr_ = DirectRPCN(aPIm, aZGr);

    //denormalize
    aPGr = NormGr(aPGr_, true);

    return( aPGr );
}


Pt3dr cRPC::DirectRPCN(const Pt2dr &aP, const double &aZ) const
{
    int aK;
    double aPoly[20];
    double aGrC1Num=0, aGrC2Num=0, aGrC1Den=0, aGrC2Den=0;
    
    const Pt3dr aPt3(aP.x, aP.y, aZ);
    CubicPolyFil(aPt3, aPoly);


    for(aK=0; aK<20; aK++)
    {
        //first ground coordinate
        aGrC1Num += aPoly[aK] * mDirSNum[aK];
        aGrC1Den += aPoly[aK] * mDirSDen[aK];

        //second ground coordinate
        aGrC2Num += aPoly[aK] * mDirLNum[aK];
        aGrC2Den += aPoly[aK] * mDirLDen[aK];

    }

    ELISE_ASSERT( !(aGrC1Den==0 || aGrC2Den==0), "Pt3dr cRPC::DirectRPCN division by 0" );
        
    return( Pt3dr( aGrC1Num/aGrC1Den,
                   aGrC2Num/aGrC2Den,
                   aZ ));
}

/* R3 -> R2 (Ground_Coords -> Sample, Line) */
Pt2dr cRPC::InverseRPC(const Pt3dr &aP) const
{
    
    Pt3dr aPGr;
    Pt2dr aPIm, aPIm_;

    //normalize
    aPGr = NormGr(aP);

    //apply inverse RPCs
    aPIm_ = InverseRPCN(aPGr);

    //denormalize
    aPIm = NormIm(aPIm_, true);

    //polynom refinement
    if(mRefine==eRP_Poly)
    {
        Pt2dr aDep = mPol->DeltaCamInit2CurIm(Pt2dr(aPIm.x, aPIm.y));
        aPIm += aDep;
    }
    return( aPIm );

}


Pt2dr cRPC::InverseRPCN(const Pt3dr &aP) const
{
    int aK;
    double aImSNum=0, aImLNum=0, aImSDen=0, aImLDen=0;
    double aPoly[20];

    CubicPolyFil(aP, aPoly);

    for(aK=0; aK<20; aK++)
    {
        aImSNum += aPoly[aK] * mInvSNum[aK];
        aImSDen += aPoly[aK] * mInvSDen[aK];

        aImLNum += aPoly[aK] * mInvLNum[aK];
        aImLDen += aPoly[aK] * mInvLDen[aK];
     
        if(DEBUG_EWELINA)
        {
            std::cout << "aPoly(" << aK << ") " << aPoly[aK] << " mInvSDen " << mInvSDen[aK] << "  s: " << aImSDen  << "\n";  
        }
    }

    if(DEBUG_EWELINA)
        std::cout << "---------aImSNum/aImSDen " << aImSNum  << " / " << aImSDen << "\n";
    
    ELISE_ASSERT(!(aImSDen==0 || aImLDen==0), "Pt3dr cRPC::InverseRPCN division by 0" );

    return( Pt2dr(aImSNum/aImSDen,
                  aImLNum/aImLDen) );

}

void cRPC::InvToDirRPC()
{
    ELISE_ASSERT(ISINV,"cRPC::InvToDirRPC(), No inverse RPC's for conversion");

    int aK, aNPt;
    Pt2dr aP;
    std::vector<Pt3dr> aGridGr, aGridIm;

    //generate a grid in normalized ground space
    GenGridNorm(mRecGrid, aGridGr);

    aNPt = int(aGridGr.size());

    //backproject to normalized image space
    for(aK=0; aK<aNPt; aK++)
    {
        aP = InverseRPCN(aGridGr.at(aK));
        aGridIm.push_back(Pt3dr(aP.x, aP.y, aGridGr.at(aK).z));
    }
    
if(0)
{
    Pt3dr aPMin, aPMax, aPSum;
    GetGridExt(aGridIm, aPMin, aPMax, aPSum);
    std::cout << "Min " << aPMin << " \n Max " << aPMax << "\n";

    GetGridExt(aGridGr, aPMin, aPMax, aPSum);
    std::cout << "Min " << aPMin << " \n Max " << aPMax << "\n";
}

    //learn the RPC parameters
    LearnParam(aGridGr, aGridIm,
                mDirSNum, mDirLNum,
                mDirSDen, mDirLDen);



    ISDIR=true;
}

/* ChSysRPC/ChSysRPC_ change the coordinate system of RPC from or to geodetic */
void cRPC::ChSysRPC(const cSystemeCoord &aChSys)
{

    std::string aTypeStr = eToString(aChSys.BSC()[0].TypeCoord());
    ELISE_ASSERT( (aTypeStr == "eTC_Proj4"), "cRPC::ChSysRPC transformation must follow the proj4 format");

    ChSysRPC_(aChSys, mRecGrid, 
               mDirSNum, mDirLNum, mDirSDen, mDirLDen,
               mInvSNum, mInvLNum, mInvSDen, mInvLDen,
               0);

}

void cRPC::ChSysRPC(const cSystemeCoord &aChSys,
               double (&aDirSNum)[20], double (&aDirLNum)[20],
               double (&aDirSDen)[20], double (&aDirLDen)[20],
               double (&aInvSNum)[20], double (&aInvLNum)[20],
               double (&aInvSDen)[20], double (&aInvLDen)[20])
{
    ChSysRPC_(aChSys, mRecGrid,
               aDirSNum, aDirLNum, aDirSDen, aDirLDen,
               aInvSNum, aInvLNum, aInvSDen, aInvLDen,
               0);
}

void cRPC::ChSysRPC_(const cSystemeCoord &aChSys, 
                      const Pt3di &aSz,
                      double (&aDirSNum)[20], double (&aDirLNum)[20],
                      double (&aDirSDen)[20], double (&aDirLDen)[20],
                      double (&aInvSNum)[20], double (&aInvLNum)[20],
                      double (&aInvSDen)[20], double (&aInvLDen)[20],
                      bool PRECISIONTEST)
{
    int aK;
    Pt3dr aP;

    cCs2Cs * aToCorSys=0;
    if(!ISMETER)
        aToCorSys = new cCs2Cs("+proj=longlat +datum=WGS84 +to " + 
                    aChSys.BSC()[0].AuxStr()[0]);
    else
    {
        aToCorSys = new cCs2Cs(" -f %.8f " + aChSys.BSC()[0].AuxStr()[0] +
                  " +to +proj=longlat +datum=WGS84 ");
    }

    vector<Pt3dr> aGridOrg, aGridCorSys, aGridOrgTest, aGridCorSysTest,
                  aGridImg, aGridImgN, aGridImgTest;

    /* Create the image grids */
    GenGridNorm(aSz,aGridImgN);
    aGridImg = NormImAll(aGridImgN,1);

    Pt3dr aPMin, aPMax, aPSum;
    GetGridExt(aGridImg, aPMin, aPMax, aPSum);
    cRPC::GenGridAbs_( Pt3dr(aPMin.x + double(0.5*(aPMax.x - aPMin.x))/aSz.x,
                             aPMin.y + double(0.5*(aPMax.y - aPMin.y))/aSz.y,
                             aPMin.z + double(0.5*(aPMax.z - aPMin.z))/aSz.z),
                       Pt3dr(aPMax.x - double(0.5*(aPMax.x - aPMin.x))/aSz.x,
                             aPMax.y - double(0.5*(aPMax.y - aPMin.y))/aSz.y,
                             aPMax.z - double(0.5*(aPMax.z - aPMin.z))/aSz.z),
                       Pt3di(aSz.x-1,aSz.y-1,aSz.z-1), aGridImgTest); 


if(0)
{
    std::cout << "ewelina check norm/unnorm" << "\n";
    Pt3dr aPMin, aPMax, aPSum;
    GetGridExt(aGridImgN, aPMin, aPMax, aPSum);
    std::cout << "Min " << aPMin << " \n Max " << aPMax << "\n";
    GetGridExt(aGridImg, aPMin, aPMax, aPSum);
    std::cout << "Min " << aPMin << " \n Max " << aPMax << "\n";

}

    /* Pass the image grid to object space grid */
    for(aK=0; aK<int(aGridImg.size()); aK++)
    {
        aP = DirectRPC(Pt2dr(aGridImg.at(aK).x,aGridImg.at(aK).y), aGridImg.at(aK).z);
        aGridOrg.push_back(aP);
    }

    for(aK=0; aK<int(aGridImgTest.size()); aK++)
    {
        aP = DirectRPC(Pt2dr(aGridImgTest.at(aK).x,aGridImgTest.at(aK).y), aGridImgTest.at(aK).z);
        aGridOrgTest.push_back(aP);
    }

    /* Transform the object grids */
    aGridCorSys  = aToCorSys->Chang(aGridOrg);
    aGridCorSysTest  = aToCorSys->Chang(aGridOrgTest);

    CalculRPC( aGridCorSys, aGridImg,
               aGridCorSysTest, aGridImgTest,
               aDirSNum, aDirLNum, aDirSDen, aDirLDen,
               aInvSNum, aInvLNum, aInvSDen, aInvLDen,
               PRECISIONTEST);
    
    
    if(ISMETER)
        ISMETER=false;
    else
        ISMETER=true;

    /* If RPC were recomputed including the polynom deformation
     * we no longer want to correct the image observations */
    if(mRefine==eRP_Poly)
        mRefine=eRP_None;
        
}


void cRPC::CalculRPC( const vector<Pt3dr> &aGridGround, 
                      const vector<Pt3dr> &aGridImg,
                      const vector<Pt3dr> &aGridGroundTest, 
                      const vector<Pt3dr> &aGridImgTest,
                      double (&aDirSNum)[20], double (&aDirLNum)[20],
                      double (&aDirSDen)[20], double (&aDirLDen)[20],
                      double (&aInvSNum)[20], double (&aInvLNum)[20],
                      double (&aInvSDen)[20], double (&aInvLDen)[20],
                      bool PRECISIONTEST)
{

    /* Update offset/scale */
    NewImOffScal(aGridImg);
    NewGrOffScal(aGridGround);

    /* Normalise */
    vector<Pt3dr> aGridImgN = NormImAll(aGridImg,0);

    /* Normalise (with updated offset/scale) */
    vector<Pt3dr> aGridGroundN = NormGrAll(aGridGround);


if(0)
{
    std::cout << "ewelina learn" << "\n";
    Pt3dr aPMin, aPMax, aPSum;
    GetGridExt(aGridImgN, aPMin, aPMax, aPSum);
    std::cout << "Min " << aPMin << " \n Max " << aPMax << "\n";

    GetGridExt(aGridGroundN, aPMin, aPMax, aPSum);
    std::cout << "Min " << aPMin << " \n Max " << aPMax << "\n";
}

    /* Learn direct RPC */
    LearnParam(aGridGroundN, aGridImgN,
                aDirSNum, aDirLNum,
                aDirSDen, aDirLDen);

    /* Learn inverse RPC */
    LearnParam(aGridImgN, aGridGroundN,
                aInvSNum, aInvLNum,
                aInvSDen, aInvLDen);


    if(PRECISIONTEST)
    {
        int aK;

        Pt2dr aPDifMoy(0,0);
        for(aK=0; aK<int(aGridGroundTest.size()); aK++)
        {
            Pt2dr aPB = InverseRPC(aGridGroundTest.at(aK));
            Pt2dr aPDif = Pt2dr(abs(aGridImgTest.at(aK).x-aPB.x),
                                abs(aGridImgTest.at(aK).y-aPB.y));


            aPDifMoy.x += aPDif.x;
            aPDifMoy.y += aPDif.y;
        }


        if( (double(aPDifMoy.x)/(aGridGroundTest.size())) > 1 || (double(aPDifMoy.y)/(aGridGroundTest.size())) > 1 )
            std::cout << "RPC recalculation"
                <<  " precision: " << double(aPDifMoy.x)/(aGridGroundTest.size()) << " "
                << double(aPDifMoy.y)/(aGridGroundTest.size()) << " [pix] \n xXXXXXXXXX ATTENTION XXXXXXXXXXXXXXXXX\n"
                <<                                                       " x          badly estimated RPCs          X\n"
                <<                                                       " x          choose a larger crop          X\n"
                <<                                                       " xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n";
    }
    
        
}

/* ground_coord_1 * mDirSDen - mDirSNum =0
 * ground_coord_2 * mDirLDen - mDirLNum =0
 * OR
 * image__coord_1* mInvSDen - mInvSNum =0
 * image__coord_2* mInvLDen - mInvLNum =0
 * <-1,1> normalized space */
void cRPC::LearnParam(std::vector<Pt3dr> &aGridIn, 
                    std::vector<Pt3dr> &aGridOut, 
                    double (&aSol1)[20], double (&aSol2)[20],
                    double (&aSol3)[20], double (&aSol4)[20])
{
    int aK, aNPts = int(aGridIn.size());
    double aC1, aC2;

    L2SysSurResol aSys1(39), aSys2(39);

    for(aK=0; aK<aNPts; aK++)
    {
        double aEq1[39];
        double aEq2[39];
    
        aC1 = aGridIn.at(aK).x;
        aC2 = aGridIn.at(aK).y;

         DifCubicPolyFil(aGridOut.at(aK), aC1, aEq1);
         DifCubicPolyFil(aGridOut.at(aK), aC2, aEq2);

         aSys1.AddEquation(1, aEq1, aC1);
         aSys2.AddEquation(1, aEq2, aC2);

    }

    bool ok;
    Im1D_REAL8 aResol1 = aSys1.GSSR_Solve(&ok);
    Im1D_REAL8 aResol2 = aSys2.GSSR_Solve(&ok);

    double* aData1 = aResol1.data();
    double* aData2 = aResol2.data();

    for(aK=0; aK<20; aK++)
    {
        aSol1[aK] = aData1[aK];
        aSol2[aK] = aData2[aK];
    }

    aSol3[0] = 1;
    aSol4[0] = 1;

    for(aK=20; aK<39; aK++)
    {
        aSol3[aK-19] = aData1[aK];
        aSol4[aK-19] = aData2[aK];
    }
}

void cRPC::SetPolyn(const std::string &aName)
{
    mPol = cPolynomial_BGC3M2D::NewFromFile(aName);
    mRefine = eRP_Poly; 
    //mPol->Show();
}

/* Even if an image crop is used, the RPC are recomputed on the original img
 * btw in [Tao & Hu, 2001] horizontal grid every ~600pix, vert grid every ~500m
 * in [Guo, 2006] empirically showed that 20x20x3 grid is sufficient */
void cRPC::SetRecGrid_(const bool &aMETER, const Pt3dr &aPMin, const Pt3dr &aPMax, Pt3di &aSz)
{
    int aSamplX, aSamplY, aSamplZ;

    //grid spacing in 3D
    int aHorizM = 500, aVertM = 100;

    if(aMETER)
    {
        aSamplX = floor((aPMax.x - aPMin.x)/aHorizM);
        aSamplY = floor((aPMax.y - aPMin.y)/aHorizM);
        aSamplZ = floor((aPMax.z - aPMin.z)/aVertM);    
    }
    else
    {
        double aFprntLonM =  6378137 *
                             (aPMax.x - aPMin.x) * M_PI /180.0;
        double aFprntLatM = 6378137 *
                            (aPMax.y - aPMin.y) * M_PI /180.0;

        aSamplX = floor(aFprntLonM/aHorizM);
        aSamplY = floor(aFprntLatM/aHorizM);
        aSamplZ = floor((aPMax.z - aPMin.z)/aVertM);
    }


    //if there is less than 5 layers in Z ([Tao & Hu, 2001] suggest min of 3)
    while (aSamplZ<4)
        aSamplZ++;

    //if planar grid smaller than 15 (to avoid bad estimations in cropped images)
    while (aSamplX<15)
        aSamplX++;
    while (aSamplY<15)
        aSamplY++;

    //if the grid does not suffice to calculate 78 coefficients of the RPCs
    while ( (aSamplX*aSamplY*aSamplZ)<80 )
        aSamplX++;

    aSz = Pt3di(aSamplX,aSamplY,aSamplZ);

    if(0)
        std::cout <<"RPC grid: " << aSz << "\n";

}


void cRPC::SetRecGrid()
{
    Pt3dr aPMin(mGrC1[0], mGrC2[0], mGrC3[0]);
    Pt3dr aPMax(mGrC1[1], mGrC2[1], mGrC3[1]);
    
    SetRecGrid_(ISMETER, aPMin, aPMax, mRecGrid);

    
}

void cRPC::GenGridNorm_(const Pt2dr aRange, const Pt3di &aSz, std::vector<Pt3dr> &aGrid)
{
    int aK1, aK2, aK3;
    double aExt = aRange.y - aRange.x;
    double aSh = aRange.y;

    double aZS = double(aExt)/aSz.z;
    double aXS = double(aExt)/aSz.x;
    double aYS = double(aExt)/aSz.y;

    for (aK1 = 0; aK1 <= aSz.x; aK1++)
    {
        for (aK2 = 0; aK2 <= aSz.y; aK2++)
        {
            for(aK3 = 0; aK3 <= aSz.z; aK3++ )
            {
                Pt3dr aPt;
                aPt.x = aK1*aXS -aSh;
                aPt.y = aK2*aYS -aSh;
                aPt.z = aK3*aZS -aSh;
                aGrid.push_back(aPt);
            }
        }
    }
}

void cRPC::GenGridNorm(const Pt3di &aSz, std::vector<Pt3dr> &aGrid)
{
    Pt2dr aRng(-1,1);
    cRPC::GenGridNorm_(aRng,aSz,aGrid);

    /*int aK1, aK2, aK3;

    double aZS = double(2)/aSz.z;
    double aXS = double(2)/aSz.x;
    double aYS = double(2)/aSz.y;

    for (aK1 = 0; aK1 <= aSz.x; aK1++)
    {
        for (aK2 = 0; aK2 <= aSz.y; aK2++)
        {
            for(aK3 = 0; aK3 <= aSz.z; aK3++ )
            {
                Pt3dr aPt;
                aPt.x = aK1*aXS -1;
                aPt.y = aK2*aYS -1;
                aPt.z = aK3*aZS -1;
                aGrid.push_back(aPt);
            }
        }
    }*/
}

void cRPC::GenGridAbs(const Pt3di &aSz, std::vector<Pt3dr> &aGrid)
{
    const Pt3dr aPMin(mGrC1[0], mGrC2[0], mGrC3[0]);
    const Pt3dr aPMax(mGrC1[1], mGrC2[1], mGrC3[1]);

    GenGridAbs_(aPMin, aPMax, aSz, aGrid);
    
}

void cRPC::GenGridAbs_(const Pt3dr &aPMin, const Pt3dr &aPMax, const Pt3di &aSz, std::vector<Pt3dr> &aGrid)
{
    int aK1, aK2, aK3;
    Pt3dr aP, aStep;


    aStep = Pt3dr(double(aPMax.x - aPMin.x)/aSz.x,
                  double(aPMax.y - aPMin.y)/aSz.y,
                  double(aPMax.z - aPMin.z)/aSz.z );


    for(aK1=0; aK1<=aSz.x; aK1++)
        for(aK2=0; aK2<=aSz.y; aK2++)
            for(aK3=0; aK3<=aSz.z; aK3++)
            {
                aP = Pt3dr( aPMin.x + aStep.x*aK1,
                            aPMin.y + aStep.y*aK2,
                            aPMin.z + aStep.z*aK3);
                
                aGrid.push_back(aP);
            }
    

}

Pt2dr cRPC::NormIm(const Pt2dr &aP, bool aDENORM) const
{
    Pt2dr aRes;

    if(aDENORM)
    {
        aRes.x = aP.x*mImScal[0] + mImOff[0];
        aRes.y = aP.y*mImScal[1] + mImOff[1];
    }
    else
    {
        aRes.x = (aP.x - mImOff[0])/mImScal[0];
        aRes.y = (aP.y - mImOff[1])/mImScal[1];
    }

    return(aRes);
}

vector<Pt3dr> cRPC::NormImAll(const vector<Pt3dr> &aP, bool aDENORM) const
{
    int aK;
    Pt2dr aPP;
    vector<Pt3dr> aRes;

    for(aK=0; aK<int(aP.size()); aK++)
    {
        aPP = NormIm(Pt2dr(aP.at(aK).x,aP.at(aK).y), aDENORM);
        aRes.push_back( Pt3dr(aPP.x, aPP.y, NormGrZ(aP.at(aK).z, aDENORM)));
    }

    return(aRes);
}

Pt3dr cRPC::NormGr(const Pt3dr &aP, bool aDENORM) const
{
   Pt3dr aRes;

   if(aDENORM)
   {
       aRes.x = aP.x*mGrScal[0] + mGrOff[0];
       aRes.y = aP.y*mGrScal[1] + mGrOff[1];
   }
   else
   {
       aRes.x = (aP.x - mGrOff[0])/mGrScal[0]; 
       aRes.y = (aP.y - mGrOff[1])/mGrScal[1]; 
   }

   aRes.z = NormGrZ(aP.z,aDENORM);

   return(aRes);
}


double cRPC::NormGrZ(const double &aZ, bool aDENORM) const
{
    double aRes;

    if(aDENORM)
        aRes = aZ*mGrScal[2] + mGrOff[2];
    else
        aRes = (aZ - mGrOff[2])/mGrScal[2];

    return(aRes);
}

vector<Pt3dr> cRPC::NormGrAll(const vector<Pt3dr> &aP, bool aDENORM) const
{
    int aK;
    vector<Pt3dr> aRes;

    for(aK=0; aK<int(aP.size()); aK++)
        aRes.push_back(NormGr(aP.at(aK), aDENORM));
    
    return(aRes);
}


void cRPC::NewImOffScal(const std::vector<Pt3dr> & aGrid)
{
   Pt3dr aExtMin, aExtMax, aSumXYZ;

   GetGridExt(aGrid,aExtMin,aExtMax,aSumXYZ);

if(0)
{
    std::cout << "ewelinaa cRPC::NewImOffScal " << "\n";
    std::cout << "Min " << aExtMin << " \n Max " << aExtMax << "\n";
}

   mImOff[0] = aExtMin.x + double(aExtMax.x - aExtMin.x)/2;
   mImOff[1] = aExtMin.y + double(aExtMax.y - aExtMin.y)/2;

   mImScal[0] =abs(aExtMax.x - mImOff[0]);
   mImScal[1] =abs(aExtMax.y - mImOff[1]);

   ReconstructValidityxy();
    
}

void cRPC::NewGrC(double &aGrC1min, double &aGrC1max,
                   double &aGrC2min, double &aGrC2max,
                   double &aGrC3min, double &aGrC3max)
{
    mGrC1[0] = aGrC1min;
    mGrC1[1] = aGrC1max;
    mGrC2[0] = aGrC2min;
    mGrC2[1] = aGrC2max;
    mGrC3[0] = aGrC3min;
    mGrC3[1] = aGrC3max;
}

void cRPC::NewGrOffScal(const std::vector<Pt3dr> & aGrid)
{
    Pt3dr aExtMin, aExtMax, aSumXYZ;

    GetGridExt(aGrid,aExtMin,aExtMax,aSumXYZ);

    mGrOff[0] = aExtMin.x + double(aExtMax.x-aExtMin.x)/2;//double(aSumXYZ.x)/aGrid.size();
    mGrOff[1] = aExtMin.y + double(aExtMax.y-aExtMin.y)/2;//double(aSumXYZ.y)/aGrid.size();
    mGrOff[2] = aExtMin.z + double(aExtMax.z-aExtMin.z)/2;//double(aSumXYZ.z)/aGrid.size();

    //std::abs(aExtMax.x - mGrOff[0]) > std::abs(aExtMin.x - mGrOff[0]) ?
        mGrScal[0] = std::abs(aExtMax.x - mGrOff[0]);// :
        //mGrScal[0] = std::abs(aExtMin.x - mGrOff[0]);

    //std::abs(aExtMax.y - mGrOff[1]) > std::abs(aExtMin.y - mGrOff[1]) ?
        mGrScal[1] = std::abs(aExtMax.y - mGrOff[1]);// :
        //mGrScal[1] = std::abs(aExtMin.y - mGrOff[1]);

    //std::abs(aExtMax.z - mGrOff[2]) > std::abs(aExtMin.x - mGrOff[2]) ?
        mGrScal[2] = std::abs(aExtMax.z - mGrOff[2]);// :
       // mGrScal[2] = std::abs(aExtMin.z - mGrOff[2]);

    
    NewGrC(aExtMin.x,aExtMax.x,
            aExtMin.y,aExtMax.y,
            aExtMin.z,aExtMax.z);
}

void cRPC::GetGridExt(const std::vector<Pt3dr> & aGrid,
                       Pt3dr & aExtMin,
                       Pt3dr & aExtMax,
                       Pt3dr & aSumXYZ )
{
    int aK;

    double aEX=0, aEY=0, aEZ=0;
    double X_min=aGrid.at(0).x, X_max=X_min,
           Y_min=aGrid.at(0).y, Y_max=Y_min,
           Z_min=aGrid.at(0).z, Z_max=Z_min;

    for(aK=0; aK<int(aGrid.size()); aK++)
    {
        aEX+=abs(aGrid.at(aK).x);
        aEY+=abs(aGrid.at(aK).y);
        aEZ+=abs(aGrid.at(aK).z);

        if(aGrid.at(aK).x > X_max)
            X_max = aGrid.at(aK).x;

        if(aGrid.at(aK).x < X_min)
            X_min = aGrid.at(aK).x;

        if(aGrid.at(aK).y < Y_min)
            Y_min = aGrid.at(aK).y;

        if(aGrid.at(aK).y > Y_max)
            Y_max = aGrid.at(aK).y;

        if(aGrid.at(aK).z < Z_min)
            Z_min = aGrid.at(aK).z;

        if(aGrid.at(aK).z > Z_max)
            Z_max = aGrid.at(aK).z;

    }

    aExtMin = Pt3dr(X_min,Y_min,Z_min);
    aExtMax = Pt3dr(X_max,Y_max,Z_max);
    aSumXYZ = Pt3dr(aEX,aEY,aEZ);

}


double cRPC::GetImRow1() const
{
    return mImRows[0];
}

double cRPC::GetImRow2() const
{
    return mImRows[1];
}

double cRPC::GetImCol1() const
{
    return mImCols[0];
}

double cRPC::GetImCol2() const
{
    return mImCols[1];
}

void cRPC::GetGrC1(vector<double>& aC) const
{
    aC.push_back(mGrC1[0]);
    aC.push_back(mGrC1[1]);
}

void cRPC::GetGrC2(vector<double>& aC) const
{
    aC.push_back(mGrC2[0]);
    aC.push_back(mGrC2[1]);
}

double cRPC::GetGrC31() const
{
    return mGrC3[0];
}

double cRPC::GetGrC32() const
{
    return mGrC3[1];
}

void cRPC::GetImOff(vector<double>& aOff) const
{
    aOff.push_back(mImOff[0]);
    aOff.push_back(mImOff[1]);
}

void cRPC::GetImScal(vector<double>& aSca) const
{
    aSca.push_back(mImScal[0]);
    aSca.push_back(mImScal[1]);
}

void cRPC::GetGrOff(vector<double>& aOff) const
{
    aOff.push_back(mGrOff[0]);
    aOff.push_back(mGrOff[1]);
    aOff.push_back(mGrOff[2]);
}

void cRPC::GetGrScal(vector<double>& aSca) const
{
    aSca.push_back(mGrScal[0]);
    aSca.push_back(mGrScal[1]);
    aSca.push_back(mGrScal[2]);
}

bool cRPC::IsDir() const
{
    return ISDIR;
}

bool cRPC::IsInv() const
{
    return ISINV;
}

/****************************************************************/
/*                Reading                                       */
/****************************************************************/

void cRPC::ReadEUCLIDIUM(const std::string &aFile)
{
    std::ifstream ASCIIfi(aFile.c_str());
    ELISE_ASSERT(ASCIIfi.good(), "cRPC::ReadEUCLIDIUM(const std::string &aFile) file not found ");

    int aC=0;
    std::string line;
    std::string delim="\t";
    {
        for(aC=0; aC<29; aC++)
            std::getline(ASCIIfi, line);


        /* Validity */
        std::vector<std::string> aSegs; 
        std::getline(ASCIIfi, line);
        
        size_t pos=0;
        while ((pos = line.find(delim)) != (std::string::npos)) 
        {
            aSegs.push_back(line.substr(0, pos));
            line.erase(0, pos + delim.length());
        }

        mGrOff[0] = atof( (aSegs.at(2)).c_str());
        mGrScal[0] = atof(line.c_str());
        
        
        aSegs.clear();
        std::getline(ASCIIfi, line);
        while ((pos = line.find(delim)) != std::string::npos) 
        {
            aSegs.push_back(line.substr(0, pos));
            line.erase(0, pos + delim.length());
        }
        
        mGrOff[1] = atof( (aSegs.at(2)).c_str());
        mGrScal[1] = atof(line.c_str());
        
        aSegs.clear();
        std::getline(ASCIIfi, line);
        while ((pos = line.find(delim)) != std::string::npos) 
        {
            aSegs.push_back(line.substr(0, pos));
            line.erase(0, pos + delim.length());
        }

        mGrOff[2] = atof( (aSegs.at(2)).c_str());
        mGrScal[2] = atof(line.c_str());

        aSegs.clear();
        std::getline(ASCIIfi, line);
        while ((pos = line.find(delim)) != std::string::npos) 
        {
            aSegs.push_back(line.substr(0, pos));
            line.erase(0, pos + delim.length());
        }
                
        mImOff[1] = atof( (aSegs.at(2)).c_str());
        mImScal[1] = atof(line.c_str());
        
        aSegs.clear();
        std::getline(ASCIIfi, line);
        while ((pos = line.find(delim)) != std::string::npos) 
        {
            aSegs.push_back(line.substr(0, pos));
            line.erase(0, pos + delim.length());
        }
                
        mImOff[0] = atof( (aSegs.at(2)).c_str());
        mImScal[0] = atof(line.c_str());
        
        /* Inverse coefficients i
         * sample */
        for(aC=0; aC<26; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }

        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());

            mInvSNum[aC] = atof(line.c_str());
        }

        for(aC=0; aC<2; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }

        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());
            
            mInvSDen[aC] = atof(line.c_str());
        }

        /* line */
        for(aC=0; aC<2; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }

        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());
            
            mInvLNum[aC] = atof(line.c_str());
        }

        for(aC=0; aC<2; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }

        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());
            
            mInvLDen[aC] = atof(line.c_str());
        }

        /* Direct coefficients */
        for(aC=0; aC<61; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }
        
        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());
            
            mDirSNum[aC] = atof(line.c_str());
        }

        for(aC=0; aC<2; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }

        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());
            
            mDirSDen[aC] = atof(line.c_str());
        }

        for(aC=0; aC<2; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }
        
        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());
            
            mDirLNum[aC] = atof(line.c_str());
        }
    
        for(aC=0; aC<2; aC++)
        {
            std::istringstream iss;
            std::getline(ASCIIfi, line);
        }
        
        for(aC=0; aC<20; aC++)
        {
            std::getline(ASCIIfi, line);
            pos = line.find(delim);
            line.erase(0, pos + delim.length());
            
            mDirLDen[aC] = atof(line.c_str());
        }
    
    }

    ISINV=true;
    ISDIR=true;
   
    ReconstructValidityxy();
    ReconstructValidityXY();
    ReconstructValidityH();
}

void cRPC::ReadASCII(const std::string &aFile)
{
    std::ifstream ASCIIfi(aFile.c_str());
    ELISE_ASSERT(ASCIIfi.good(), "cRPC::ReadASCII(const std::string &aFile) ASCII file not found ");

    std::string line;
    std::string a, b;
    int aC;


    //Line Offset
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mImOff[1] >> b;
    }

    //Samp Offset
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mImOff[0] >> b;
    }

    //Lat Offset
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mGrOff[1] >> b;
    }

    //Lon Offset
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mGrOff[0] >> b;
    }

    //Height Offset
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mGrOff[2] >> b;
    }

    //Line Scale
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mImScal[1] >> b;
    }
    

    //Sample Scale
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mImScal[0] >> b;
    }


    //Lat Scale
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mGrScal[1] >> b;
    }

    //Lon Scale
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mGrScal[0] >> b;
    }

    
    //Height Scale
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mGrScal[2] >> b;
    }

    //inverse line num
    for(aC=0; aC<20; aC++)
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mInvLNum[aC];
    }

    //inverse line den
    for(aC=0; aC<20; aC++)
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mInvLDen[aC];
    }

    //inverse sample num
    for(aC=0; aC<20; aC++)
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mInvSNum[aC];
    }

    //inverse sample den
    for(aC=0; aC<20; aC++)
    {
        std::istringstream iss;
        std::getline(ASCIIfi, line);
        iss.str(line);
        iss >> a >> mInvSDen[aC];
    }

    ISINV=true;

    ReconstructValidityxy();
    ReconstructValidityXY();
    ReconstructValidityH();
  

}

int cRPC::ReadASCIIMeta(const std::string &aMeta, const std::string &aFile)
{
    std::string errorcmd = "cRPC::ReadASCIIMeta; rename your metadata file to: " + aMeta;
    std::ifstream MetaFi(aMeta.c_str());
    ELISE_ASSERT(MetaFi.good(), errorcmd.c_str());

    bool aMetaIsFound=false;

    std::string line=" ";
    std::string a, b, c, d;
    std::vector<double> avLat, avLon;

    std::string aToMatchOne = "Product";
    std::string aToMatchTwo = "Metadata";
    std::string aToMatchThree = "Component";
    std::string aToMatchFour = "File";
    std::string aToMatchFive = "Name:";
    std::string aToMatchSix = "Columns:";
    std::string aToMatchSev = "Coordinate:";


    while(MetaFi.good())
    {
		std::getline(MetaFi, line);
		std::istringstream iss;
		iss.str(line);
		iss >> a >> b >> c;
		if( a==aToMatchOne &&
	    	b==aToMatchThree &&
	    	c==aToMatchTwo )
		{
	    	std::getline(MetaFi, line);
	    	std::istringstream iss2;    
	    	iss2.str(line);
	    	iss2 >> a >> b >> c >> d;
	
	    	while(MetaFi.good())
	    	{
				//iterate to line "Component File Name:"
	        	if( !((a==aToMatchThree) &&
                     (b==aToMatchFour) &&
                     (c==aToMatchFive)))
	        	{
		    		std::getline(MetaFi, line);
		    		std::istringstream iss3;
		    		iss3.str(line);
		    		iss3 >> a >> b >> c >> d;
	        	}
				else
				{

		    		//check if the filenames correspond
		    		if(d.substr(0,d.length()-4)==aFile.substr(0,aFile.length()-4))
		    		{

						while(MetaFi.good())
						{

			    			//find
						// the Columns and Rows
						// the coords of the corners
			    			std::getline(MetaFi, line);
				    		std::istringstream iss4;
			    			iss4.str(line);
			    			iss4 >> a >> b >> c;


			    			//columns
			    			if(a==aToMatchSix)
			    			{
			        			mImCols[0]=0;
                    			mImCols[1]=std::atof(b.c_str())-1;	
			    
			        			//rows
			        			std::getline(MetaFi, line);
			        			std::istringstream iss5;
			        			iss5.str(line);
			        			iss5 >> a >> b >> c;

			        			mImRows[0]=0;
			        			mImRows[1]=std::atof(b.c_str())-1;

								aMetaIsFound=true;

								MetaFi.close();

								return EXIT_SUCCESS;
			    			}
						else if(a==aToMatchSev)
						{
						    //corner1
						    std::getline(MetaFi, line);
						    {std::istringstream issl0;
						    issl0.str(line);
						    issl0 >> a >> b >> c;}
                                                    std::cout << b << std::endl;
							
						    avLat.push_back(std::atof(b.c_str()));
						    
						    std::getline(MetaFi, line);
						    {std::istringstream issl0;
						    issl0.str(line);
						    issl0 >> a >> b >> c;}
                                                    std::cout << b << std::endl;
                                                   
						    avLon.push_back(std::atof(b.c_str()));
						    
						    //corner2 
						    std::getline(MetaFi, line); 
						    std::getline(MetaFi, line);
						    {std::istringstream issl0;
						    issl0.str(line);
						    issl0 >> a >> b >> c;}
                                                    std::cout << b << std::endl;

						    avLat.push_back(std::atof(b.c_str()));

						    std::getline(MetaFi, line);
						    {std::istringstream issl0;
						    issl0.str(line);
						    issl0 >> a >> b >> c;}
                                                    std::cout << b << std::endl;

						    avLon.push_back(std::atof(b.c_str()));

						    //corner3
						    std::getline(MetaFi, line);
						    std::getline(MetaFi, line);
					            {std::istringstream issl0;
					            issl0.str(line);
					            issl0 >> a >> b >> c;}
                                                    std::cout << b << std::endl;

					            avLat.push_back(std::atof(b.c_str()));

						    std::getline(MetaFi, line);
					            {std::istringstream issl0;
					            issl0.str(line);
					            issl0 >> a >> b >> c;}
                                                    std::cout << b << std::endl;

					            avLon.push_back(std::atof(b.c_str()));	    
                                              
						    //corner4
						    std::getline(MetaFi, line);
						    std::getline(MetaFi, line);
						    {std::istringstream issl0;
					            issl0.str(line);
                                                    issl0 >> a >> b >> c;}
						    std::cout << b << std::endl;

						    avLat.push_back(std::atof(b.c_str()));

						    std::getline(MetaFi, line);
						    {std::istringstream issl0;
					            issl0.str(line);
						    issl0 >> a >> b >> c;}
						    std::cout << b << std::endl;

						    avLon.push_back(std::atof(b.c_str()));


						    mGrC1[0] = *std::min_element(avLon.begin(),avLon.end());
						    mGrC1[1] = *std::max_element(avLon.begin(),avLon.end());

						    mGrC2[0] = *std::min_element(avLat.begin(),avLat.end()); 
						    mGrC2[1]  = *std::max_element(avLat.begin(),avLat.end());


						    
						}
						}
		    		}
		    		else
		    		{
		        		std::getline(MetaFi, line);
						std::istringstream iss6;
						iss6.str(line);
						iss6 >> a >> b >> c >> d;

		    		}
				}
	    	}
		}

    }
    MetaFi.close();

    ELISE_ASSERT(!aMetaIsFound, " no metadata found in cRPC::ReadASCIIMetaData");

    return EXIT_FAILURE;   
}

void cRPC::ReadXML(const std::string &aFile)
{
    cElXMLTree aTree(aFile.c_str());
    cElXMLTree* aNodes;

    {
        aNodes = aTree.GetUnique(std::string("NUMROWS"));
        mImRows[0] = 0;
        mImRows[1] = std::atof(aNodes->GetUniqueVal().c_str());
    }

    {
        aNodes = aTree.GetUnique(std::string("NUMCOLUMNS"));
        mImCols[0] = 0;
        mImCols[1] = std::atof(aNodes->GetUniqueVal().c_str());
    }

    {
        aNodes = aTree.GetUnique(std::string("SAMPOFFSET"));
        mImOff[0] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    {
        aNodes = aTree.GetUnique(std::string("LINEOFFSET"));
        mImOff[1] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    
    {
        aNodes = aTree.GetUnique(std::string("LONGOFFSET"));
        mGrOff[0] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    {
        aNodes = aTree.GetUnique(std::string("LATOFFSET"));
        mGrOff[1] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    {
        aNodes = aTree.GetUnique(std::string("HEIGHTOFFSET"));
        mGrOff[2] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }


    {
        aNodes = aTree.GetUnique(std::string("SAMPSCALE"));
        mImScal[0] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    {
        aNodes = aTree.GetUnique(std::string("LINESCALE"));
        mImScal[1] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    {
        aNodes = aTree.GetUnique(std::string("LONGSCALE"));
        mGrScal[0] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    {
        aNodes = aTree.GetUnique(std::string("LATSCALE"));
        mGrScal[1] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    {
        aNodes = aTree.GetUnique(std::string("HEIGHTSCALE"));
        mGrScal[2] = std::atof(aNodes->GetUniqueVal().c_str()); 
    }

    aNodes = aTree.GetUnique(std::string("LINENUMCOEF"));
    {
        std::istringstream aSs;
        aSs.str(aNodes->GetUniqueVal());

        aSs >> mInvLNum[0] >> mInvLNum[1] >> mInvLNum[2]
            >> mInvLNum[3] >> mInvLNum[4] >> mInvLNum[5]
            >> mInvLNum[6] >> mInvLNum[7] >> mInvLNum[8]
            >> mInvLNum[9] >> mInvLNum[10] >> mInvLNum[11]
            >> mInvLNum[12] >> mInvLNum[13] >> mInvLNum[14]
            >> mInvLNum[15] >> mInvLNum[16] >> mInvLNum[17]
            >> mInvLNum[18] >> mInvLNum[19];
    }

    aNodes = aTree.GetUnique(std::string("LINEDENCOEF"));
    {
        std::istringstream aSs;
        aSs.str(aNodes->GetUniqueVal());

        aSs >> mInvLDen[0] >> mInvLDen[1] >> mInvLDen[2]
            >> mInvLDen[3] >> mInvLDen[4] >> mInvLDen[5]
            >> mInvLDen[6] >> mInvLDen[7] >> mInvLDen[8]
            >> mInvLDen[9] >> mInvLDen[10] >> mInvLDen[11]
            >> mInvLDen[12] >> mInvLDen[13] >> mInvLDen[14]
            >> mInvLDen[15] >> mInvLDen[16] >> mInvLDen[17]
            >> mInvLDen[18] >> mInvLDen[19];

    }

    aNodes = aTree.GetUnique(std::string("SAMPNUMCOEF"));
    {
        std::istringstream aSs;
        aSs.str(aNodes->GetUniqueVal());

        aSs >> mInvSNum[0] >> mInvSNum[1] >> mInvSNum[2]
            >> mInvSNum[3] >> mInvSNum[4] >> mInvSNum[5]
            >> mInvSNum[6] >> mInvSNum[7] >> mInvSNum[8]
            >> mInvSNum[9] >> mInvSNum[10] >> mInvSNum[11]
            >> mInvSNum[12] >> mInvSNum[13] >> mInvSNum[14]
            >> mInvSNum[15] >> mInvSNum[16] >> mInvSNum[17]
            >> mInvSNum[18] >> mInvSNum[19];

    }

    aNodes = aTree.GetUnique(std::string("SAMPDENCOEF"));
    {
        std::istringstream aSs;
        aSs.str(aNodes->GetUniqueVal());

        aSs >> mInvSDen[0] >> mInvSDen[1] >> mInvSDen[2]
            >> mInvSDen[3] >> mInvSDen[4] >> mInvSDen[5]
            >> mInvSDen[6] >> mInvSDen[7] >> mInvSDen[8]
            >> mInvSDen[9] >> mInvSDen[10] >> mInvSDen[11]
            >> mInvSDen[12] >> mInvSDen[13] >> mInvSDen[14]
            >> mInvSDen[15] >> mInvSDen[16] >> mInvSDen[17]
            >> mInvSDen[18] >> mInvSDen[19];
        
    }

    cElXMLTree* aNodesFilOne;
    std::vector<double> aLongMM, aLatMM;

    aNodes = aTree.GetUnique(std::string("BAND_P"));
    aNodesFilOne = aNodes->GetUnique("ULLON");
    aLongMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    aNodesFilOne = aNodes->GetUnique("URLON");
    aLongMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    aNodesFilOne = aNodes->GetUnique("LRLON");
    aLongMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    aNodesFilOne = aNodes->GetUnique("LLLON");
    aLongMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    mGrC1[0] = *std::min_element(aLongMM.begin(),aLongMM.end());
    mGrC1[1] = *std::max_element(aLongMM.begin(),aLongMM.end());

    
    aNodesFilOne = aNodes->GetUnique("ULLAT");
    aLatMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    aNodesFilOne = aNodes->GetUnique("URLAT");
    aLatMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    aNodesFilOne = aNodes->GetUnique("LRLAT");
    aLatMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    aNodesFilOne = aNodes->GetUnique("LLLAT");
    aLatMM.push_back(std::atof((aNodesFilOne->GetUniqueVal()).c_str()));

    mGrC2[0] = *std::min_element(aLatMM.begin(),aLatMM.end());
    mGrC2[1] = *std::max_element(aLatMM.begin(),aLatMM.end());

    ISINV=true;
    
    ReconstructValidityH();


}


void cRPC::ReadDimap(const std::string &aFile)
{
    int aK;
    cElXMLTree aTree(aFile.c_str());


    std::list<cElXMLTree*>::iterator aIt;

    std::string aSNumStr;
    std::string aSDenStr;
    std::string aLNumStr;
    std::string aLDenStr;

    {
        std::list<cElXMLTree*> aNoeuds = aTree.GetAll(std::string("Direct_Model"));


        for (aK=1; aK<21; aK++)
        {
            aSNumStr = "SAMP_NUM_COEFF_";
            aSDenStr = "SAMP_DEN_COEFF_";
            aLNumStr = "LINE_NUM_COEFF_";
            aLDenStr = "LINE_DEN_COEFF_";

            aSNumStr = aSNumStr + ToString(aK);
            aSDenStr = aSDenStr + ToString(aK);
            aLNumStr = aLNumStr+ ToString(aK);
            aLDenStr = aLDenStr+ ToString(aK);

            for(aIt=aNoeuds.begin(); aIt!=aNoeuds.end(); aIt++)
            {

                mDirSNum[aK-1] = std::atof((*aIt)->GetUnique(aSNumStr.c_str())->GetUniqueVal().c_str());
                mDirSDen[aK-1] = std::atof((*aIt)->GetUnique(aSDenStr.c_str())->GetUniqueVal().c_str());
                mDirLNum[aK-1] = std::atof((*aIt)->GetUnique(aLNumStr.c_str())->GetUniqueVal().c_str());
                mDirLDen[aK-1] = std::atof((*aIt)->GetUnique(aLDenStr.c_str())->GetUniqueVal().c_str());
            }
        }
    }

    {
        std::list<cElXMLTree*> aNoeudsInv = aTree.GetAll(std::string("Inverse_Model"));

        for (aK=1; aK<21; aK++)
        {
            
            aSNumStr = "SAMP_NUM_COEFF_";
            aSDenStr = "SAMP_DEN_COEFF_";
            aLNumStr = "LINE_NUM_COEFF_";
            aLDenStr = "LINE_DEN_COEFF_";

            aSNumStr = aSNumStr + ToString(aK);
            aSDenStr = aSDenStr + ToString(aK);
            aLNumStr = aLNumStr+ ToString(aK);
            aLDenStr = aLDenStr+ ToString(aK);

            for(aIt=aNoeudsInv.begin(); aIt!=aNoeudsInv.end(); aIt++)
            {

                mInvSNum[aK-1] = std::atof((*aIt)->GetUnique(aSNumStr.c_str())->GetUniqueVal().c_str());
                mInvSDen[aK-1] = std::atof((*aIt)->GetUnique(aSDenStr.c_str())->GetUniqueVal().c_str());
                mInvLNum[aK-1] = std::atof((*aIt)->GetUnique(aLNumStr.c_str())->GetUniqueVal().c_str());
                mInvLDen[aK-1] = std::atof((*aIt)->GetUnique(aLDenStr.c_str())->GetUniqueVal().c_str());
            }
        }
        
    }

    {
        std::list<cElXMLTree*> aNoeudsRFM = aTree.GetAll(std::string("RFM_Validity"));

        
        {
            for(aIt=aNoeudsRFM.begin(); aIt!=aNoeudsRFM.end(); aIt++)
            {
                mImScal[0] = std::atof((*aIt)->GetUnique("SAMP_SCALE")->GetUniqueVal().c_str());
                mImScal[1] = std::atof((*aIt)->GetUnique("LINE_SCALE")->GetUniqueVal().c_str());
                mImOff[0] = std::atof((*aIt)->GetUnique("SAMP_OFF")->GetUniqueVal().c_str());
                mImOff[1] = std::atof((*aIt)->GetUnique("LINE_OFF")->GetUniqueVal().c_str());

                mGrScal[0] = std::atof((*aIt)->GetUnique("LONG_SCALE")->GetUniqueVal().c_str());
                mGrScal[1] = std::atof((*aIt)->GetUnique("LAT_SCALE")->GetUniqueVal().c_str());
                mGrScal[2] = std::atof((*aIt)->GetUnique("HEIGHT_SCALE")->GetUniqueVal().c_str());
                
                mGrOff[0] = std::atof((*aIt)->GetUnique("LONG_OFF")->GetUniqueVal().c_str());
                mGrOff[1] = std::atof((*aIt)->GetUnique("LAT_OFF")->GetUniqueVal().c_str());
                mGrOff[2] = std::atof((*aIt)->GetUnique("HEIGHT_OFF")->GetUniqueVal().c_str());
            }
                
        //}

        //{
          //  std::list<cElXMLTree*> aNoeudsDirVal = aTree.GetAll(std::string("Direct_Model_Validity_Domain"));

            for(aIt=aNoeudsRFM.begin(); aIt!=aNoeudsRFM.end(); aIt++)
            {
                mImRows[0] = std::atof((*aIt)->GetUnique("FIRST_ROW")->GetUniqueVal().c_str());
                mImRows[1] = std::atof((*aIt)->GetUnique("LAST_ROW")->GetUniqueVal().c_str());
                mImCols[0] = std::atof((*aIt)->GetUnique("FIRST_COL")->GetUniqueVal().c_str());
                mImCols[1] = std::atof((*aIt)->GetUnique("LAST_COL")->GetUniqueVal().c_str());
            }
        //}

        //{
          //  std::list<cElXMLTree*> aNoeudsInvVal = aTree.GetAll(std::string("Inverse_Model_Validity_Domain"));

            for(aIt=aNoeudsRFM.begin(); aIt!=aNoeudsRFM.end(); aIt++)
            {
                mGrC1[0] = std::atof((*aIt)->GetUnique("FIRST_LON")->GetUniqueVal().c_str());
                mGrC1[1] = std::atof((*aIt)->GetUnique("LAST_LON")->GetUniqueVal().c_str());
                mGrC2[0] = std::atof((*aIt)->GetUnique("FIRST_LAT")->GetUniqueVal().c_str());
                mGrC2[1] = std::atof((*aIt)->GetUnique("LAST_LAT")->GetUniqueVal().c_str());
            }
        }
    }
    ISDIR=true;
    ISINV=true;

    ReconstructValidityH();

    
}

void cRPC::ReconstructValidityxy()
{
    ELISE_ASSERT(ISINV, "cRPC::ReconstructValidityxy() RPCs need to be initialised" );
    
    mImRows[0] = -1 * mImScal[1] + mImOff[1];
    mImRows[1] = 1 * mImScal[1] + mImOff[1];

    mImCols[0] = -1 * mImScal[0] + mImOff[0];
    mImCols[1] = 1 * mImScal[0] + mImOff[0];

}

void cRPC::ReconstructValidityXY()
{
    ELISE_ASSERT(ISINV, "cRPC::ReconstructValidityXY() RPCs need to be initialised" );

    mGrC1[0] = -1 * mGrScal[0] + mGrOff[0];
    mGrC1[1] =  1 * mGrScal[0] + mGrOff[0];

    mGrC2[0] = -1 * mGrScal[1] + mGrOff[1];
    mGrC2[1] =  1 * mGrScal[1] + mGrOff[1];

}

void cRPC::ReconstructValidityH()
{
    ELISE_ASSERT(ISINV, "cRPC::ReconstructValidityH() RPCs need to be initialised" );

    mGrC3[0] = -1 * mGrScal[2] + mGrOff[2];
    mGrC3[1] =  1 * mGrScal[2] + mGrOff[2];

}

void cRPC::Show()
{
    std::cout << "RPC:" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "long_scale   : " << mGrScal[0] << " | long_off   : " << mGrOff[0] << std::endl;
    std::cout << "lat_scale    : " << mGrScal[1] << " | lat_off    : " << mGrOff[1] << std::endl;
    std::cout << "height_scale    : " << mGrScal[2] << " | height_off    : " << mGrOff[2] << std::endl;
    std::cout << "samp_scale   : " << mImScal[0] << " | samp_off   : " << mImOff[0] << std::endl;
    std::cout << "line_scale   : " << mImScal[1] << " | line_off   : " << mImOff[1] << std::endl;
    std::cout << "first_row    : " << mImRows[0] << " | last_row   : " << mImRows[1] << std::endl;
    std::cout << "first_col    : " << mImCols[0] << " | last_col   : " << mImCols[1] << std::endl;
    std::cout << "first_lon    : " << mGrC1[0] << " | last_lon   : " << mGrC1[1]  << std::endl;
    std::cout << "first_lat    : " << mGrC2[0] << " | last_lat   : " << mGrC2[1] << std::endl;
    std::cout << "first_height    : " << mGrC3[0] << " | last_height   : " << mGrC3[1] << std::endl;


    std::cout << "direct sample num : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mDirSNum[aK] << " ";

    std::cout << "\n";
    std::cout << "direct sample den : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mDirSDen[aK] << " ";

    std::cout << "\n";
    std::cout << "direct line num : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mDirLNum[aK] << " ";

    std::cout << "\n";
    std::cout << "direct line den : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mDirLDen[aK] << " ";



    std::cout << "\n-------------------";
    std::cout << "\ninverse sample num : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mInvSNum[aK] << " ";

    std::cout << "\n";
    std::cout << "inverse sample den : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mInvSDen[aK] << " ";

    std::cout << "\n";
    std::cout << "inverse line num : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mInvLNum[aK] << " ";

    std::cout << "\n";
    std::cout << "inverse line den : \n";
    for(int aK=0; aK<20; aK++)
        std::cout << mInvLDen[aK] << " ";

    
    std::cout << "\n";
}

Pt3di cRPC::GetGrid() const
{
    return mRecGrid;
}

bool cRPC::IsMetric() const
{
    return ISMETER;
}

int Grid2RPC_main(int argc,char ** argv)
{
    cInterfChantierNameManipulateur * aICNM;
    std::string aName, aSeulName, aNameOrient, aChSysStr,
                aDir, aDest="ER-Tmp", aGBName;
    std::string aCom1, aCom2;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aName,"Name of Image")
                   << EAMC(aNameOrient,"Name of input Orientation File")
                   << EAMC(aDest,"Directory of output Orientation (MyDir -> Oi-MyDir)"),
        LArgMain()
                   << EAM(aChSysStr,"ChSys", true, "Change coordinate file (MicMac XML convention)")
    );

    SplitDirAndFile(aDir, aSeulName, aName);

    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    
    
    aCom1 = MM3dBinFile_quotes("Convert2GenBundle") + " " + 
           aName + " " + aNameOrient + " " + 
           aDest + " ChSys=" + aChSysStr + 
           " Degre=0";   
    std::cout << "aCom1 " << aCom1 << "\n";

    TopSystem(aCom1.c_str());
    
    
    
    
    aGBName = aICNM->StdNameCamGenOfNames(aDest, aSeulName);
    std::cout << "aGBName " << aGBName << " aSeulName "  << aSeulName << "\n";

    aCom2 = MM3dBinFile_quotes("SateLib RecalRPC") + " " + aGBName;
    std::cout << "aCom2 " << aCom2 << "\n";

    TopSystem(aCom2.c_str());
            
    
    ELISE_fp::RmFile(aGBName); 

    return EXIT_SUCCESS;
}

int CropRPC_main(int argc,char ** argv)
{
    cInterfChantierNameManipulateur * aICNM;
    std::string aCropName,aFullName;
    std::string aDir, aDirSav="Crop";
    std::string aName;
    std::list<std::string> aListFile;
    Pt2dr aO(100,100), aSz(10000,10000);

    ElInitArgMain
        (
         argc, argv,
         LArgMain() << EAMC(aCropName,"Orientation file of the image defining the crop extent (in cXml_CamGenPolBundle format)")
                    << EAMC(aFullName,"Pattern of orientation files to be cropped accordingly (in cXml_CamGenPolBundle format)")
                    << EAMC(aDirSav, "Directory of output orientation files"),
         LArgMain() << EAM(aO,"Org", true, "Origin of the rectangular crop; Def=[100,100]")
                    << EAM(aSz,"Sz", true, "Size of the crop; Def=[10000,10000]")
        );

    SplitDirAndFile(aDir, aName, aFullName);
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aName);

    /* Retrieve 3D volume corresponding to the crop */
    std::vector<Pt3dr> aBounds;

    CameraRPC aCamCrop(aCropName);
    const double aZ = aCamCrop.GetAltiSol(); 
    const Pt2dr a0(aO.x,aO.y);
    const Pt2dr a1(a0.x+aSz.x,a0.y);
    const Pt2dr a2(a0.x+aSz.x,a0.y+aSz.y);
    const Pt2dr a3(a0.x,a0.y+aSz.y);

    aBounds.push_back(aCamCrop.ImEtZ2Terrain(a0,aZ));
    aBounds.push_back(aCamCrop.ImEtZ2Terrain(a1,aZ));
    aBounds.push_back(aCamCrop.ImEtZ2Terrain(a2,aZ));
    aBounds.push_back(aCamCrop.ImEtZ2Terrain(a3,aZ));

    
    //for every image : 
    //  read
    //  backproject the 3D bounds and define 2D bounds (this will be the cut & 2d grid bounds)
    //  create grids in 2d and 3d and pass to CalculRPC
    //  save IMG + RPC (in original coords)

    std::list<std::string>::iterator itL=aListFile.begin();
    for( ; itL !=aListFile.end(); itL++ )
    {

        CameraRPC aCamCropTMP(aDir + (*itL));
        aCamCropTMP.CropRPC(aDirSav, aDir + (*itL), aBounds);
    }


    return EXIT_SUCCESS;
}

int RecalRPC_main(int argc,char ** argv)
{
    cInterfChantierNameManipulateur * aICNM;
    std::string aFullName;
    std::string aDir;
    std::string aName;
    std::list<std::string> aListFile;

    bool aVf=false;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName,"Orientation file (or pattern) in cXml_CamGenPolBundle format"),
        LArgMain() << EAM(aVf,"Vf", "Verification of the re-calculation on all tie points (Def = false)")
     );

    SplitDirAndFile(aDir, aName, aFullName);
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aName);

    std::list<std::string>::iterator itL=aListFile.begin();
    for( ; itL !=aListFile.end(); itL++ )
    {
        cRPC::Save2XmlStdMMName(aDir + (*itL),"");
    }
    
   
    //- in euclid (geod can be added)
    if(aVf)
    {

        Pt3di aSz(100,100,100);

        itL=aListFile.begin();
        for( ; itL !=aListFile.end(); itL++ )
        {
            std::cout << (*itL) <<"\n";
            
            /* In euclid sys 
             * -read cameras (input with the poly & recalculated) */
            CameraRPC aCamIni(aDir + (*itL));
            CameraRPC aCamRec(cRPC::NameSave(aDir + (*itL)));
       
            /* -verify independently */
            cRPCVerf aVfIni(aCamIni,aSz);
            cRPCVerf aVfRec(aCamRec,aSz);
        
            aVfIni.Do();
            aVfRec.Do(aVfIni.mGrid3dFP);

            /* -compare image coordinates & object coordinates */
            aVfIni.Compare2D(aVfRec.mGrid2dBP);
            aVfIni.Compare3D(aVfRec.mGrid3dFP);
        }
    }



    return EXIT_SUCCESS;
}


// mm3d TestPbRPC Ori-RPC/GB-Orientation-S6P--2014042116840914CP.tif.xml 
void OneTestCamRPC(CameraRPC & aCam ,const Pt3dr & aP)
{
   std::cout << "P=" << aP << "Proj="  << aCam.Ter2Capteur(aP) << "\n";
}
int TestCamRPC(int argc,char** argv)
{
   std::string aName;
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aName,"Name camera"),
        LArgMain()
   );

   double anAlti = 4400;
   CameraRPC aCam(aName,anAlti);
   Pt3dr aP0 (355936.0,3127508.8,6571.52);//6571.52
   Pt2dr aStepIn(12.8,-12.8);

   DEBUG_EWELINA = true;

   for (int anX=-1 ; anX<=1 ; anX++)
   {
       for (int anY=-1 ; anY<=1 ; anY++)
       {
            OneTestCamRPC(aCam,aP0 + Pt3dr(aStepIn.x*anX,aStepIn.y*anY,0.0));
       }
   }

   return EXIT_SUCCESS;
}

/******************************************************/
/*                                                    */
/*                    cRPCVerf                        */
/*                                                    */
/******************************************************/

cRPCVerf::cRPCVerf(const CameraRPC &aCam, const Pt3di &aSz) :
    mSz(aSz),
    mCam(&aCam)
{}

void cRPCVerf::Do(const std::vector<Pt3dr> &aG)
{
    ELISE_ASSERT((int(aG.size())==0 || int(aG.size())==(mSz.x*mSz.y*mSz.z)),"cRPCVerf::Do(); incoherent grid size"); 

    int aK1;
    int aGSz = mSz.x*mSz.y*mSz.z;
    Pt2dr aRange(-0.9,-0.9);//verification will take place on 80% of the validity space
    Pt3dr aPt;
   
    //to be able to access some class functions
    cRPC aRPC = mCam->GetRPCCpy();
    
    if( aG.size() == 0)
    {
        std::vector<Pt3dr> aGrid2dN,aGrid2d;
        
        cRPC::GenGridNorm_(aRange, mSz, aGrid2dN);
        aGrid2d = aRPC.NormImAll(aGrid2dN,1);
    

        //forward project the points
        for(aK1=0; aK1<aGSz; aK1++)
            mGrid3d.push_back(mCam->ImEtZ2Terrain(
                              Pt2dr(aGrid2d.at(aK1).x,aGrid2d.at(aK1).y),
                              aGrid2d.at(aK1).z));

    }
    else
        mGrid3d = aG;


    //backproject all points of the grid to image space
    for(aK1=0; aK1<aGSz; aK1++)
        mGrid2dBP.push_back(mCam->Ter2Capteur(mGrid3d.at(aK1)));
    
    //forward project to object space
    double aDistPlanMoy=0, aDistAltMoy=0;
    double aDistPlanMax=0,  aDistAltMax=0;
    double aTmp1, aTmp2;

    for(aK1=0; aK1<int(mGrid2dBP.size()); aK1++)
    {
        mGrid3dFP.push_back(mCam->ImEtZ2Terrain(
                            Pt2dr(mGrid2dBP.at(aK1).x,mGrid2dBP.at(aK1).y), 
                            mGrid3d.at(aK1).z));
        aTmp1 = sqrt(std::pow(abs(mGrid3dFP.at(aK1).x - mGrid3d.at(aK1).x),2) +
                     std::pow(abs(mGrid3dFP.at(aK1).y - mGrid3d.at(aK1).y),2));
        aTmp2 = abs(mGrid3dFP.at(aK1).z - mGrid3d.at(aK1).z);

        if( aDistPlanMax<aTmp1 ) aDistPlanMax=aTmp1;
        if( aDistAltMax<aTmp2 ) aDistAltMax=aTmp2;

        aDistPlanMoy += aTmp1;
        aDistAltMoy  += aTmp2;

    }

if(0)
    std::cout << "---\n" 
              << "RPC TEST IN 3D" << "\n"
              << "plan moy =" << aDistPlanMoy/aGSz << ", max=" << aDistPlanMax << "\n" 
              << "alt moy  =" << aDistAltMoy/aGSz <<  ", max=" << aDistAltMax << "\n"
              << "UNIT METRIC=" << aRPC.IsMetric() << ", DEGREE=" << !(aRPC.IsMetric()) << "\n";


}

void cRPCVerf::Compare2D(std::vector<Pt2dr> &aGrid2d) const
{
   ELISE_ASSERT(aGrid2d.size()==mGrid2dBP.size(),"cRPCVerf::Compare2D ; incoherent grid size to compare to"); 
    std::cout << "\n---\nRPC COMAPRISON IN 2D (before and after Recal)" << "\n";

    int aK=0;
    int aCnt=int(mGrid2dBP.size());
    Pt2dr aMoy(0,0);
    Pt2dr aMax(0,0);
    double aTmp1, aTmp2;
    int aRep=10, aRepCnt=0;


    for(aK=0; aK<aCnt; aK++)
    {
        aTmp1 = abs(aGrid2d.at(aK).x - mGrid2dBP.at(aK).x);
        aTmp2 = abs(aGrid2d.at(aK).y - mGrid2dBP.at(aK).y);

        aMoy.x += aTmp1;
        aMoy.y += aTmp2;

        if(aTmp1>aMax.x) aMax.x=aTmp1;
        if(aTmp2>aMax.y) aMax.y=aTmp2;

        if(aTmp1>aRep || aTmp2>aRep)
        {
            //std::cout << "(" << aTmp1 << "," << aTmp2 << "):" << mGrid2dBP.at(aK) << " ";
            aRepCnt++;
        }
    }


    std::cout << "row moy =" << aMoy.x/aCnt << ", max=" << aMax.x << "\n"
              << "col moy =" << aMoy.y/aCnt << ", max=" << aMax.y << "\n"
              << "discrepancies>" << aRep << "->" << aRepCnt << " out of " << aCnt << "\n" ;
}

void cRPCVerf::Compare3D(std::vector<Pt3dr> &aGrid3d) const
{
   ELISE_ASSERT(aGrid3d.size()==mGrid3dFP.size(),"cRPCVerf::Compare3D ; incoherent grid size to compare to"); 
    std::cout << "\n---\nRPC COMAPRISON IN 3D  (before and after Recal)" << "\n";

    int aK=0;
    int aCnt=int(mGrid3dFP.size());
    Pt3dr aMoy(0,0,0);
    Pt3dr aMax(0,0,0);
    double aTmp1, aTmp2, aTmp3;
    int aRep=10, aRepCnt=0;

    for(aK=0; aK<aCnt; aK++)
    {
        aTmp1 = abs(aGrid3d.at(aK).x - mGrid3dFP.at(aK).x);
        aTmp2 = abs(aGrid3d.at(aK).y - mGrid3dFP.at(aK).y);
        aTmp3 = abs(aGrid3d.at(aK).z - mGrid3dFP.at(aK).z);

        aMoy.x += aTmp1;
        aMoy.y += aTmp2;
        aMoy.z += aTmp3;

        if(aTmp1>aMax.x) aMax.x=aTmp1;
        if(aTmp2>aMax.y) aMax.y=aTmp2;
        if(aTmp3>aMax.z) aMax.z=aTmp3;

        if(aTmp1>aRep || aTmp2>aRep)
        {
            //std::cout << "(" << aTmp1 << "," << aTmp2 << "):" << mGrid3dFP.at(aK) << " ";
            aRepCnt++;
        }
    }

    //to be able to say the units
    const cRPC * aRPC = mCam->GetRPC();
    
    std::cout << "X moy =" << aMoy.x/aCnt << ", max=" << aMax.x << "\n"
              << "Y moy =" << aMoy.y/aCnt << ", max=" << aMax.y << "\n"
              << "Z moy =" << aMoy.z/aCnt << ", max=" << aMax.z << "\n"
              << "discrepancies>" << aRep << "->" << aRepCnt << " out of " << aCnt << "\n" 
              << "UNIT METRIC=" << aRPC->IsMetric() << ", DEGREE=" << !(aRPC->IsMetric()) << "\n";

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

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

CameraRPC::CameraRPC(std::string const &aNameFile, const  std::string &aModeRPC, const Pt2di &aGridSz) :
       mGridSz(aGridSz)	
{
   
   mRPC = new RPC();
   
   if (aModeRPC=="PLEIADE" || aModeRPC=="SPOT")
       mRPC->ReadDimap(aNameFile);
   else if(aModeRPC=="QUICKBIRD" || aModeRPC=="WORLDVIEW" )
       mRPC->ReadRPB(aNameFile);
   else if(aModeRPC=="IKONOS" || aModeRPC=="CARTOSAT")
   {
       mRPC->ReadASCII(aNameFile);
       mRPC->InverseToDirectRPC(mGridSz);
   }
   else {ELISE_ASSERT(false,"Unknown RPC mode");}

  // mRPC->info();

}


Pt2dr CameraRPC::Ter2Capteur(const Pt3dr & aP) const
{
    
    return Pt2dr(0,0);
}

bool CameraRPC::PIsVisibleInImage   (const Pt3dr & aP) const
{
    return   true;//mCam->PIsVisibleInImage (aP);
}

ElSeg3D  CameraRPC::Capteur2RayTer(const Pt2dr & aP) const
{
    //AssertCamInit();   
    //assert that mRPC are there   

    Pt3dr aP1RayL3(aP.x, aP.y, mRPC->height_off+100), //on mean ground+100
	  aP2RayL3(aP.x, aP.y, mRPC->height_off);//on mean ground

    return F2toRayonLPH(aP1RayL3, aP2RayL3);
}

ElSeg3D CameraRPC::F2toRayonLPH(Pt3dr &aP0,Pt3dr & aP1) const
{
    return(ElSeg3D(mRPC->DirectRPC(aP0), mRPC->DirectRPC(aP1)));
}

bool   CameraRPC::HasRoughCapteur2Terrain() const
{
    return true;
}

bool  CameraRPC::HasPreciseCapteur2Terrain() const
{
    return true;
}

Pt3dr CameraRPC::RoughCapteur2Terrain   (const Pt2dr & aP) const
{
    return Pt3dr(0,0,0);//PtOfIndexInterpol(aP);
}

Pt3dr CameraRPC::PreciseCapteur2Terrain   (const Pt2dr & aP) const
{
    return Pt3dr(0,0,0);//PtOfIndexInterpol(aP);
}

double CameraRPC::ResolSolOfPt(const Pt3dr & aP) const
{
    //Pt2dr aPIm = Ter2Capteur(aP);
    //ElSeg3D aSeg = Capteur2RayTer(aPIm+Pt2dr(1,0));
    return 1.0;//aSeg.DistDoite(aP);
}

double CameraRPC::ResolSolGlob() const
{
    return(0.0);
}

bool  CameraRPC::CaptHasData(const Pt2dr & aP) const
{
    return true;//IndexHasContenuForInterpol(aP);
}

Pt2dr CameraRPC::ImRef2Capteur   (const Pt2dr & aP) const
{
    return Pt2dr(0,0);//aP / mParams.SsResolRef().Val();
}

double  CameraRPC::ResolImRefFromCapteur() const
{
    return  0.0;//mParams.SsResolRef().Val();
}

bool CameraRPC::IsP1P2IsAltitude() const
{
    //returns true if the height_off [km] inside RPCs is within a range
    //otherwise regarded as invalid information 
    return ((mRPC->height_off < 10000) && (mRPC->height_off > -10000));
}

Pt2di CameraRPC::SzBasicCapt3D() const
{
    return  (Pt2di(mRPC->last_row - mRPC->first_row,
 	           mRPC->last_col - mRPC->first_col));
}

/* Export to xml following the cXml_ScanLineSensor standard 
 * - first  iter - generate the bundle grid in geodetic coordinate system (CS) and
 *                 convert to desired CS
 * - second iter - export to xml */
void CameraRPC::ExpImp2Bundle(const std::string & aSysOut, 
		              const std::string & aName, 
			      std::vector<std::vector<ElSeg3D> > aGridToExp) const
{
        //Check that the direct RPC exists
        ELISE_ASSERT(mRPC->IS_DIR_INI,"No direct RPC's in CameraRPC::ExpImp2Bundle");	

	Pt2dr aGridStep = Pt2dr( double(SzBasicCapt3D().x)/mGridSz.x ,
			         double(SzBasicCapt3D().y)/mGridSz.y );

	std::string aDirTmp = "csconv";
	std::string aFiPrefix = "Bundle_";

	std::string aLPHFiTmp = aDirTmp + "/" + aFiPrefix + aName  + "_LPH_CS.txt";
	std::string aXYZFiTmp = aDirTmp + "/" + aFiPrefix + aName  + "_XYZ_CS.txt";
	std::string aXMLFiTmp = aFiPrefix + aName  + ".xml";

	int aGr=0, aGc=0;
	if(aGridToExp.size()==0)
	{
		ELISE_fp::MkDirSvp(aDirTmp);
		std::ofstream aFO(aLPHFiTmp.c_str());
		aFO << std::setprecision(15);
	
		//create the bundle grid in geodetic CS & save	
		ElSeg3D aSegTmp(Pt3dr(0,0,0),Pt3dr(0,0,0));
		for( aGr=0; aGr<mGridSz.x; aGr++ )
			for( aGc=0; aGc<mGridSz.y; aGc++ )
			{

				aSegTmp = Capteur2RayTer( Pt2dr(aGr*aGridStep.x,aGc*aGridStep.y) );
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

		aGridToExp.resize(mGridSz.x);
		int aCntTmp=0;
		for( aGr=0; aGr<mGridSz.x; aGr++ )
			for( aGc=0; aGc<mGridSz.y; aGc++ )
			{
				aGridToExp.at(aGr).push_back ( ElSeg3D(aPtsTmp.at(aCntTmp), 
						  	               aPtsTmp.at(aCntTmp+1)) );
				aCntTmp++;
				aCntTmp++;

			}

		ExpImp2Bundle(aSysOut, aName, aGridToExp);
	}
	else
	{

		cXml_ScanLineSensor aSLS;

		aSLS.P1P2IsAltitude() = IsP1P2IsAltitude();
		aSLS.LineImIsScanLine() = true;
		aSLS.GroundSystemIsEuclid() = true;    

		aSLS.ImSz() = SzBasicCapt3D();

		aSLS.StepGrid() = aGridStep;

		aSLS.GridSz() = mGridSz;	

		for( aGr=0; aGr<mGridSz.x; aGr++ )
		{
			cXml_OneLineSLS aOL;
			aOL.IndLine() = aGr;
			for( aGc=0; aGc<mGridSz.y; aGc++ )
			{
				cXml_SLSRay aOR;
				aOR.IndCol() = aGc;

				aOR.P1() = aGridToExp.at(aGr).at(aGc).P0();
				aOR.P2() = aGridToExp.at(aGr).at(aGc).P1();

				aOL.Rays().push_back(aOR);

			}
			aSLS.Lines().push_back(aOL);
		}
		//export to XML format
		MakeFileXML(aSLS, aXMLFiTmp);
	}		    
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

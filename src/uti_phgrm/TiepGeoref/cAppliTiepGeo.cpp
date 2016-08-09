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


#include "TiepGeo.h"



/**********************************************************************/
/*                                                                    */
/*                            cAppliTiepGeo                           */
/*                                                                    */
/**********************************************************************/

cAppliTiepGeo::cAppliTiepGeo(int argc,char **argv) :
    mMasterImStr(""),
	mCor(0.6),
    mZoom0(32),
	mNum(1)
{

    
	ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(mPatImage, "Pattern of images",  eSAM_IsPatFile)
                        << EAMC(mOri,"Orientation directory", eSAM_IsDir),
            LArgMain()  << EAM(mZoom0, "Zoom0", true, "Zoom init, pow of 2, (Def 32)")
                        << EAM(mCor, "Cor", true, "Corelation threshold (def 0.6)")
         );


    mDir = DirOfFile(mPatImage);
	//correct names ble -> Ori-ble
	if (EAMIsInit(&mOri))
	{
		StdCorrecNameOrient(mOri,mDir);
	}

    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    cElemAppliSetFile anEASF(mPatImage);
    mFilesIm = anEASF.SetIm();




    //pairse the images
	std::string aName;
	Pt2dr aPInf( 1E50, 1E50);
	Pt2dr aPSup(-1E50,-1E50);
	int aImSom = mFilesIm->size();

    std::cout << "Images:" << aImSom << "\n";
    for(int aK=0; aK<aImSom; aK++)
	{
		aName = mICNM->StdNameCamGenOfNames(mOri,mFilesIm->at(aK));
		std::cout << aK << " " << aName << "\n";

		cImageTiepGeo * aC = new cImageTiepGeo(aName);
		aC->SetNum(aK);

		mVIm.push_back( aC );
		mMapIm[aName] = aC;

		//footprint
		const Box2dr aBox = mVIm.at(aK)->BoxSol();
		aPInf = Inf(aBox._p0,aPInf);
		aPSup = Sup(aBox._p1,aPSup);	
    }
	mGlobBox = Box2dr(aPInf,aPSup);




	//connectivity graph
	std::cout << "Connectivity graph:" << "\n";
	for (int aK1=0 ; aK1<aImSom ; aK1++)
	{
		for (int aK2=aK1+1 ; aK2<aImSom ; aK2++)
		{
			if (mVIm[aK1]->HasInter(*(mVIm[aK2])))
			{
				std::cout << aK1 << "->" << aK2 << "\n";
				cLnk2ImTiepGeo *aLnk = new cLnk2ImTiepGeo(mVIm[aK1], mVIm[aK2],
											LocPxFileMatch(mDir, mNum, mZoom0),
											LocPx2FileMatch(mDir, mNum, mZoom0));
				AddLnk(aLnk);
			}
		}
	}
	std::vector<cLnk2ImTiepGeo *> aV0(mVIm.size(),0);
	mVVLnk = std::vector<std::vector<cLnk2ImTiepGeo *> >(mVIm.size(),aV0);
	for (std::list<cLnk2ImTiepGeo *>::const_iterator itL = mLnk2Im.begin(); itL!=mLnk2Im.end() ; itL++)
	{
		cImageTiepGeo & aI1 = (*itL)->Im1();
		cImageTiepGeo & aI2 = (*itL)->Im2();

		mVVLnk[aI1.Num()][aI2.Num()] = *itL;
	}



	//do master
	DoMaster();
	
}

//this must be implemented, 
//for the moment I take the first image
void cAppliTiepGeo::DoMaster()
{
	mMasterImStr = mMapIm.begin()->first;
	mMasterIm    = mMapIm[mMasterImStr];
}

void cAppliTiepGeo::AddLnk(cLnk2ImTiepGeo *aLnk)
{
	mLnk2Im.push_back(aLnk);
}

void cAppliTiepGeo::DoPx1Px2()
{
	int aK;
	for(aK=0; aK<int(mLnk2Im.size()); aK++)
	{
		
	}

		
}

void cAppliTiepGeo::DoTapioca()
{}

void cAppliTiepGeo::Exe()
{
    DoPx1Px2();

    DoTapioca();
}

int TiepGeoref_main(int argc,char **argv)
{
    cAppliTiepGeo * anAppli = new cAppliTiepGeo(argc,argv);

    anAppli->Exe();

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
footer-MicMac-eLiSe-25/06/2007*/

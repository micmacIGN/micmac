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


/*
*/

#include "StdAfx.h"

/************************************************************/
/*                                                          */
/*                   cBlockEtal                             */
/*                                                          */
/************************************************************/

void cBlockEtal::AddEtal(const cParamEtal & aParam)
{
   cEtalonnage *pEt = new cEtalonnage(mIsLastEtape,aParam,this);

   pEt->InitNormDrad(cEtalonnage::TheNameDradFinale);

   if (pE0 == 0)
      pE0 = pEt;
   mEtals.push_back(pEt);
}

cBlockEtal::cBlockEtal(bool isLastEtape,const cParamEtal & aParam,bool AddAll) :
    mIsLastEtape (isLastEtape),
    mSet    (aParam.TypeSysResolve()),
    mParam  (aParam),
    pE0     (0)
{
	if (AddAll)
	{
            AddEtal(aParam);

	    const std::vector<std::string>& vEt = aParam.CamRattachees();
	    for 
            (
                std::vector<std::string>::const_iterator iE = vEt.begin();
		iE != vEt.end();
		iE++
            )
                AddEtal(cParamEtal::FromStr(*iE));
	}
}

cSetEqFormelles & cBlockEtal::Set()
{
   return mSet;
}


void cEtalonnage::TestEpip
     (
          Video_Win aW,
	  Pt2dr aP,
	  Pt2dr P2,
	  const std::string & aName,
	  CpleEpipolaireCoord* aCpl, 
	  bool   Im1,
	  bool   show
      )
{
     REAL Zoom = 1;
     aW = aW.chc(Pt2dr(0,0),Pt2dr(Zoom,Zoom));
     Pt2di aDec = Pt2di(aP - aW.sz()/(2*Zoom));
     ELISE_COPY
     (
        aW.all_pts(),
	// Tiff_Im(aName.c_str()).in(),
        trans(Tiff_Im(aName.c_str()).in(0),aDec),
        aW.ogray()
     );
     cout << "DEC = " << aDec << " SZ " << aW.sz() << "\n";

     aW.draw_circle_loc(aP-Pt2dr(aDec),2.0,aW.pdisc()(P8COL::red));

     REAL Step = 50;
     for (REAL Pax = -3000; Pax <= 3000 ; Pax += Step)
     {
	     Pt2dr aQ1 = aCpl->Homol(P2,Pt2dr(Pax,0),!Im1);
	     if (show && (ElAbs(Pax<75)))
             {
                 cout << P2 << " " << Pax << " = " << aQ1 << "\n";
             }
	     Pt2dr aQ2 = aCpl->Homol(P2,Pt2dr(Pax+Step,0),!Im1);

	     aW.draw_seg(aQ1-Pt2dr(aDec),aQ2-Pt2dr(aDec),aW.pdisc()(P8COL::blue));

	     aQ1 = aCpl->Homol(P2,Pt2dr(Pax,-20),!Im1);
	     aQ2 = aCpl->Homol(P2,Pt2dr(Pax,20),!Im1);
	     aW.draw_seg(aQ1-Pt2dr(aDec),aQ2-Pt2dr(aDec),aW.pdisc()(P8COL::green));

     }
}

void cEtalonnage::WriteCamDRad(const std::string & aShortName)
{
     std::string aName = mParam.Directory() + aShortName + ".eor";
     ELISE_fp aFile(aName.c_str(),ELISE_fp::WRITE);
     pCamDRad->write(aFile);
     aFile.close();
}

static std::vector<double> NoParAdd;


void cBlockEtal::TestMultiRot(INT K1,INT K2)
{
     cout << "SIZ " << mEtals.size() << " " << K1 << " " << K2 << "\n";
     cEtalonnage * E1 = AT(mEtals,K1) ;  // .at(K1);
     cEtalonnage * E2 = AT(mEtals,K2) ;  // .at(K2);

     const std::vector<std::string>&VC=mParam.AllImagesCibles();
     ElRotation3D  R0 (Pt3dr(0,0,0),0,0,0);

     std::string Ref = AT(VC,2);
     ElRotation3D R1 = E1->GetBestRotEstim(Ref);
     ElRotation3D R2 = E2->GetBestRotEstim(Ref);
     R0 = R1 * R2.inv();


     for 
     (
          std::vector<std::string>::const_iterator  itC = VC.begin();
	  itC != VC.end();
          itC++
     )
     {
	     ElRotation3D R1 = E1->GetBestRotEstim(*itC);
	     ElRotation3D R2 = E2->GetBestRotEstim(*itC);
	     // Matrice de passage R2 vers R1
	     ElRotation3D R = R1 * R2.inv();
	     ElRotation3D RDif = R0 * R.inv();
	     cout << "Name " << *itC 
		     << R.tr()
		  << " TR = " << euclid(R.tr()-R0.tr())
		  << " ANGLES = " 
		  << RDif.teta01() << " "
		  << RDif.teta02() << " "
		  << RDif.teta12() << " "
		  << "\n";
     }

     cout << "COPt" << R0.ImAff(Pt3dr(0,0,0)) << "\n";
     cout << "OX : " << R0.ImVect(Pt3dr(1.0,0.0,0.0)) << "\n";
     cout << "OY : " << R0.ImVect(Pt3dr(0.0,1.0,0.0)) << "\n";
     cout << "OZ : " << R0.ImVect(Pt3dr(0.0,0.0,1.0)) << "\n";

     ElRotation3D RInv = R0.inv();
     cout << "COPt" << RInv.ImAff(Pt3dr(0,0,0)) << "\n";
     cout << "OX : " << RInv.ImVect(Pt3dr(1.0,0.0,0.0)) << "\n";
     cout << "OY : " << RInv.ImVect(Pt3dr(0.0,1.0,0.0)) << "\n";
     cout << "OZ : " << RInv.ImVect(Pt3dr(0.0,0.0,1.0)) << "\n";

     // 2291 2502 1169 2551
     if (true)
     {
         CamStenope & CR1 = *(E1->pCamDRad);
         CamStenope & CR2 = *(E2->pCamDRad);
         CR2.SetOrientation(RInv);

	 E1->WriteCamDRad(E1->Param().NameCamera()+ E2->Param().NameCamera());
	 E2->WriteCamDRad(E2->Param().NameCamera()+ E1->Param().NameCamera());
	 CpleEpipolaireCoord * aCpl=0;
	 CpleEpipolaireCoord * S_aCpl=0;

	 REAL SCALE = 4.4;

	 Video_Win W1 = Video_Win::WStd(Pt2di(100,100),5.0);
	 Video_Win W2 = Video_Win::WStd(Pt2di(100,100),5.0);
	 
	 //bool first = true;
         while (true)
         {
            CamStenopeIdeale Cam1(true,1.0,Pt2dr(0,0),NoParAdd);
            CamStenopeIdeale Cam2(true,1.0,Pt2dr(0,0),NoParAdd);
	    Cam2.SetOrientation(RInv);

	    // BONS 
	    // 2964 3444  1921 3437
	    // 2467 2030 1188 2043
	    //
	    //   BOF
	    //   2420 2023 2746 2042
	    //
	    // 2420 2023 2746 2042
            // Pt2dr p1(2420,2023),p2(2746,2042);
            Pt2dr p1(2467,2030),p2(1188,2043);

	    // if ( !first)
	        std::cin >>  p1.x >>  p1.y >>  p2.x >>  p2.y ;
            if (aCpl==0)
            {
                aCpl = CpleEpipolaireCoord::CamEpipolaire
                       (
                           CR1,p1,CR2,p2,1.0
                       );
		// S_aCpl = aCpl->MapingChScale(1/SCALE);
                S_aCpl = CpleEpipolaireCoord::CamEpipolaire
                       (
                           CR1,p1,CR2,p2,1/SCALE
                       );
            }


	    Pt2dr pN1 =  E1->ToPN(p1);
	    Pt2dr pN2 =  E2->ToPN(p2);

	    cout << pN1 << pN2 << "\n";

	    // ElSeg3D S1 = Cam1.F2toRayonR3(pN1);
	    // ElSeg3D S2 = Cam2.F2toRayonR3(pN2);
	    
	    ElSeg3D S1 = CR1.F2toRayonR3(p1);
	    ElSeg3D S2 = CR2.F2toRayonR3(p2);

	    Pt3dr Q = S1.PseudoInter(S2);

	    cout << Q << S1.DistDoite(Q)  << " " << S2.DistDoite(Q) << "\n";

	    Pt2dr PtE1 = aCpl->EPI1().Direct(p1);
	    Pt2dr PtE2 = aCpl->EPI2().Direct(p2);
	    cout << "EPIP = " << PtE1 << " " << PtE2 << "\n";
	    cout << "INV EPIP = " << aCpl->EPI1().Inverse(PtE1) 
		           << " " << aCpl->EPI2().Inverse(PtE2) << "\n";

	    Pt2dr S_p1 = p1/SCALE;
	    Pt2dr S_p2 = p2/SCALE;
	    Pt2dr S_PtE1 = S_aCpl->EPI1().Direct(S_p1);
	    Pt2dr S_PtE2 = S_aCpl->EPI2().Direct(S_p2);
	    cout << "S_EPIP = " << S_PtE1 << " " << S_PtE2 << "\n";


	    cout << p1 << aCpl->Hom21(p2,Pt2dr(0,0)) << "\n";
	    cout << p2 << aCpl->Hom12(p1,Pt2dr(0,0)) << "\n";

	    const char * NQ = "/data1/FacadesIGN/Correl/Tmp/QueueReduc1.tif";
	    const char * NT = "/data1/FacadesIGN/Correl/Tmp/TeteReduc1.tif";
	    E1->TestEpip(W1,p1,p2,NQ,aCpl,true,false);
	    E2->TestEpip(W2,p2,p1,NT,aCpl,false,false);
	    //first = false;
         }
     }
}

void cBlockEtal::TestMultiRot()
{
	TestMultiRot(0,1);
}

void cBlockEtal::TheTestMultiRot(bool isLastEtape,int argc,char ** argv)
{
    cParamEtal   aParam(argc,argv);
    cBlockEtal  aBlock(true,aParam,true);
    aBlock.TestMultiRot(); 
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

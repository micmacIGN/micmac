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



/**********************************************************************/
/*                                                                    */
/*                         cGridIncImageMnt                           */
/*                                                                    */
/**********************************************************************/

cGridIncImageMnt * cGridIncImageMnt::StdFromFile(const std::string &  aPref)
{
    Tiff_Im aFileI1 = Tiff_Im::BasicConvStd(aPref+"I1.tif");
    Tiff_Im aFileI2 = Tiff_Im::BasicConvStd(aPref+"I2.tif");

    Pt2di aSz1 = aFileI1.sz();
    Pt2di aSz2 = aFileI2.sz();

    return  new cGridIncImageMnt
	    (
	        aSz1,
		Tiff_Im::BasicConvStd(aPref+"CurZ.tif").in(),
		Virgule(
			Tiff_Im::BasicConvStd(aPref+"XHomZ0.tif").in(),
			Tiff_Im::BasicConvStd(aPref+"YHomZ0.tif").in()
		),
		Virgule(
			Tiff_Im::BasicConvStd(aPref+"Hom_dXdZ.tif").in(),
			Tiff_Im::BasicConvStd(aPref+"Hom_dYdZ.tif").in()
		 ),
		aFileI1.in(),
		aSz2,
		aFileI2.in()
	    );
}

cGridIncImageMnt::cGridIncImageMnt
(
       Pt2di    aSzI1,
       Fonc_Num fZCur,
       Fonc_Num fHom,
       Fonc_Num fDzHom,
       Fonc_Num fI1,
       Pt2di    aSzI2,
       Fonc_Num fI2
)   :
    mSzI1     (aSzI1),

    mCurZ     (aSzI1.x,aSzI1.y),
    mDCurZ    (mCurZ.data()),

    mXHomZ0   (aSzI1.x,aSzI1.y),
    mDXHomZ0  (mXHomZ0.data()),

    mYHomZ0   (aSzI1.x,aSzI1.y),
    mDYHomZ0  (mYHomZ0.data()),

    mHom_dXdZ   (aSzI1.x,aSzI1.y),
    mDHom_dXdZ  (mHom_dXdZ.data()),

    mHom_dYdZ (aSzI1.x,aSzI1.y),
    mDHom_dYdZ (mHom_dYdZ.data()),

    mIm1      (aSzI1.x,aSzI1.y),
    mDIm1     (mIm1.data()),

    mSzI2     (aSzI2),
    mIm2      (aSzI2.x,aSzI2.y) ,
    mDIm2     (mIm2.data()),

    mI2G1     (1,1),
    mDi2g1Dz  (1,1),
    mOK       (1,1),

    pSetIncs  (0),
    mNumsZInc (1,1)

{
     ELISE_COPY(All1(),fZCur,mCurZ.out());
     ELISE_COPY(All1(),fHom,Virgule(mXHomZ0.out(),mYHomZ0.out()));
     ELISE_COPY(All1(),fDzHom,Virgule(mHom_dXdZ.out(),mHom_dYdZ.out()));
     ELISE_COPY(All1(),fI1,mIm1.out());
     ELISE_COPY(All2(),fI2,mIm2.out());
}

   // ===================================================

void cGridIncImageMnt::SetEps(REAL anEps)
{
     ELISE_ASSERT(pSetIncs!=0,"cGridIncImageMnt::SetEps");
     // pSetIncs->FQC()->SetEpsB(0) ;
     // pSetIncs->FQC()->SetEpsRel(anEps);
}

void cGridIncImageMnt::InitIncs(REAL anEps)
{
     if (pSetIncs==0)
     {
        pSetIncs = new cSetEqFormelles(cNameSpaceEqF::eSysCreuxFixe);


        pRegD1  = cElCompiledFonc::RegulD1(pSetIncs,false);
        pRegD2  = cElCompiledFonc::RegulD2(pSetIncs,false);
        pRapCur = cElCompiledFonc::FoncSetVar(pSetIncs,0,false);


        mNumsZInc.Resize(mSzI1);
        mDNZI = mNumsZInc.data();

        for (INT aY=0 ; aY<mSzI1.y ; aY++)
            for (INT aX=0 ; aX<mSzI1.x ; aX++)
            {
                mDNZI[aY][aX] = pSetIncs->Alloc().NewInc("Mnt","x"+ToString(aX)+":y"+ToString(aY),&mDCurZ[aY][aX]);
            }

        pSetIncs->SetClosed();
        // pSetIncs->FQC()->SetEpsB(0) ;
        // pSetIncs->FQC()->SetEpsRel(anEps);
     }

}

void cGridIncImageMnt::SetSizeGeom()
{
    mI2G1.Resize(mSzI1);
    mDi2g1Dz.Resize(mSzI1);
    mOK.Resize(mSzI1);

    mDI2G1 = mI2G1.data();
    mDDi2g1Dz = mDi2g1Dz.data();
    mDOK = mOK.data();
}

void cGridIncImageMnt::CalcGeomI1()
{
     SetSizeGeom();
     for (INT aX=0 ; aX<mSzI1.x ; aX++)
     {
         for (INT aY=0 ; aY<mSzI1.y ; aY++)
	 {
             Pt2di aP1(aX,aY);
             Pt2dr aP2 = HomOfCurZ(aP1);
	     if (TheCIK.OkForInterp (mSzI2,aP2))
	     {
                 Pt3dr aGXY  = TheCIK.BicubValueAndDer(mDIm2,aP2);
		 mDI2G1[aY][aX] = (float) aGXY.z;
		 Pt2dr aGz = DerZ(aP1);
		 mDDi2g1Dz[aY][aX] = (float) (aGXY.x*aGz.x + aGXY.y*aGz.y);
		 mDOK[aY][aX] = 1;
	     }
	     else
	     {
		 mDOK[aY][aX] = 0;
	     }
	 }
     }
}

REAL cGridIncImageMnt::CurZ(Pt2di aP) const  
{
    return mDCurZ[aP.y][aP.x];
}
REAL cGridIncImageMnt::XHomZ0(Pt2di aP) const  
{
    return mDXHomZ0[aP.y][aP.x];
}
REAL cGridIncImageMnt::YHomZ0(Pt2di aP) const  
{
    return mDYHomZ0[aP.y][aP.x];
}
REAL cGridIncImageMnt::Hom_dXdZ(Pt2di aP) const  
{
    return mDHom_dXdZ[aP.y][aP.x];
}
REAL cGridIncImageMnt::Hom_dYdZ(Pt2di aP) const  
{
    return mDHom_dYdZ[aP.y][aP.x];
}
     // ========
Pt2dr cGridIncImageMnt::HomOfZ0(Pt2di aP) const
{
   return Pt2dr(XHomZ0(aP),YHomZ0(aP));
}
Pt2dr cGridIncImageMnt::DerZ(Pt2di aP) const
{
   return Pt2dr(Hom_dXdZ(aP),Hom_dYdZ(aP));
}
Pt2dr cGridIncImageMnt::HomOfZ(Pt2di aP,REAL aZ) const
{
   return HomOfZ0(aP) + DerZ(aP)*aZ;
}
Pt2dr cGridIncImageMnt::HomOfCurZ(Pt2di aP) const
{
   return HomOfZ(aP,CurZ(aP));
}

   // ===================================================

void cGridIncImageMnt::SauvAll(const std::string & aPrefix)
{
     Tiff_Im::CreateFromIm(mIm1    , aPrefix+"_I1.tif");
     Tiff_Im::CreateFromIm(mCurZ   , aPrefix+"_CurZ.tif");
     Tiff_Im::CreateFromIm(mXHomZ0 , aPrefix+"_XHomZ0.tif");
     Tiff_Im::CreateFromIm(mYHomZ0 , aPrefix+"_YHomZ0.tif");

     Tiff_Im::CreateFromIm(mHom_dXdZ , aPrefix+"_Hom_dXdZ.tif");
     Tiff_Im::CreateFromIm(mHom_dYdZ , aPrefix+"_Hom_dYdZ.tif");
     Tiff_Im::CreateFromIm(mIm2      , aPrefix+"_I2.tif");

     Tiff_Im::Create8BFromFonc(aPrefix+"_I2GeoI1.tif",mSzI1,fI2GeoI1());
     Tiff_Im::Create8BFromFonc(aPrefix+"_SupI1I2.tif",mSzI1,fSupI1I2());

     Tiff_Im::Create8BFromFonc
     (  aPrefix+"_CDN.tif",
         mSzI1,
         255*(round_ni((mCurZ.in()/2.0)) & 2)
     );
}

   // ===================================================

void cGridIncImageMnt::OneStepRegulD2 (REAL aPds)
{
   InitIncs(1e-6);
   std::vector<INT> mIncs;
   INT aMarge=1;
   bool first = true;
   for (INT anYC = aMarge ; anYC<mSzI1.y-aMarge ; anYC++)
   {
       for (INT anXC = aMarge ; anXC<mSzI1.x-aMarge ; anXC++)
       {
	    mIncs.clear();
            for (INT aY= anYC -aMarge ; aY<=(anYC+aMarge); aY++)
            {
                for (INT aX= anXC -aMarge ; aX<=(anXC+aMarge); aX++)
		{
	            mIncs.push_back(mDNZI[aY][aX]);
		}
	    }
	    if (first)
	    {
               first = false;
               pSetIncs->FQC()->SetOffsets(mIncs);
	    }
            pSetIncs->AddEqIndexeToSys(pRegD2,aPds/4,mIncs);
       }
   }
}

void cGridIncImageMnt::OneStepRegulD1 (REAL aPds)
{
   InitIncs(1e-6);
   std::vector<INT> mIncs;
   INT aMarge=1;
   bool first = true;
   for (INT anYC = 0 ; anYC<mSzI1.y-aMarge ; anYC++)
   {
       for (INT anXC = 0 ; anXC<mSzI1.x-aMarge ; anXC++)
       {
	    mIncs.clear();
	    mIncs.push_back(mDNZI[anYC][anXC+1]);
	    mIncs.push_back(mDNZI[anYC][anXC]);
	    mIncs.push_back(mDNZI[anYC+1][anXC]);
	    if (first)
	    {
               first = false;
               pSetIncs->FQC()->SetOffsets(mIncs);
	    }
            pSetIncs->AddEqIndexeToSys(pRegD1,aPds/2,mIncs);
       }
   }
}


void  cGridIncImageMnt::SolveSys()
{
      pSetIncs->SolveResetUpdate();
}

void cGridIncImageMnt::OneStepRapCur(REAL aPds)
{
   InitIncs(1e-6);
   std::vector<INT> mIncs;
   double * AdrRap = pRapCur->FoncSetVarAdr();
   bool first = true;
   for (INT anYC = 0 ; anYC<mSzI1.y ; anYC++)
   {
       for (INT anXC = 0 ; anXC<mSzI1.x ; anXC++)
       {
	    mIncs.clear();
	    mIncs.push_back(mDNZI[anYC][anXC]);
	    *AdrRap = mDCurZ[anYC][anXC];
	    if (first)
	    {
               first = false;
               pSetIncs->FQC()->SetOffsets(mIncs);
	    }
            pSetIncs->AddEqIndexeToSys(pRapCur,aPds,mIncs);
       }
   }
}


void cGridIncImageMnt::OneStepEqCorrel
     (
	  REAL   aPds,
          INT    aSzVgn,
	  bool   Im2Var,
	  INT    anEcart
     )
{
   INT aNbPix = ElSquare(1+2*aSzVgn);
   InitIncs(1e-6);
   cEqCorrelGrid * pECG = pSetIncs->ReuseEqCorrelGrid(aNbPix,Im2Var);
   INT aMarge = anEcart*aSzVgn;
   CalcGeomI1();
   aPds /= aNbPix;

   std::vector<INT> mIncs;
   bool first = true;
   INT aPer = mSzI1.y /10;
   for (INT anYC = aMarge ; anYC<mSzI1.y-aMarge ; anYC++)
   {
       if ((anYC % aPer) == aPer/2)
           cout << "Doing Line " << anYC << "\n";
       for (INT anXC = aMarge ; anXC<mSzI1.x-aMarge ; anXC++)
       {
            INT aK = 0;
	    mIncs.clear();
            for (INT aY= anYC -aMarge ; aY<=(anYC+aMarge); aY+= anEcart)
            {
                for (INT aX= anXC -aMarge ; aX<=(anXC+aMarge); aX+= anEcart)
		{
	            mIncs.push_back(mDNZI[aY][aX]);
                    cElemEqCorrelGrid & anElem = pECG->KthElem(aK);
		    REAL aCurZ = mDCurZ[aY][aX];
		    REAL aDerG2Z = mDDi2g1Dz[aY][aX];
                    *(anElem.mAdrGr1) = mDIm1[aY][aX];
                    *(anElem.mAdrGr2Of0) =mDI2G1[aY][aX] - aDerG2Z*aCurZ;
                    *(anElem.mAdrDGr2Dz) = mDDi2g1Dz[aY][aX];
		    if (anElem.mAdrZCur)
                       *(anElem.mAdrZCur) = aCurZ;
		    aK++;
		}
            }
	    if (first)
	    {
               first = false;
               pSetIncs->FQC()->SetOffsets(mIncs);
	    }
            pSetIncs->AddEqIndexeToSys(pECG->Fctr(),aPds,mIncs);
	    /*{
                const std::vector<double> &   V = pECG->Fctr()->Vals();
		cout << "VALS ";
	        for (INT aK=0 ; aK<INT(V.size()) ; aK++)
                   cout << V[aK] << " ";
		cout << "\n";
	    }*/
       }
   }
}


   // ===================================================


Fonc_Num  cGridIncImageMnt::fHomOfZ0()
{
    return Virgule(mXHomZ0.in(),mYHomZ0.in());
}

Fonc_Num  cGridIncImageMnt::fDerZ()
{
    return Virgule(mHom_dXdZ.in(),mHom_dYdZ.in());
}

Fonc_Num  cGridIncImageMnt::fHomOfZ(Fonc_Num fZ)
{
    return fHomOfZ0() + fDerZ() * fZ;
}

Fonc_Num  cGridIncImageMnt::fHomOfCurZ()
{
    return fHomOfZ(mCurZ.in());
}

Fonc_Num cGridIncImageMnt::fI2GeoI1()
{
   return mIm2.in(0)[fHomOfCurZ()];
}

Fonc_Num  cGridIncImageMnt::fSupI1I2()
{
    return Virgule(mIm1.in(),mIm1.in(),fI2GeoI1());
}

Flux_Pts cGridIncImageMnt::All1()
{
    return mCurZ.all_pts();
}

Flux_Pts cGridIncImageMnt::All2()
{
    return mIm2.all_pts();
}

Im2D_REAL8 cGridIncImageMnt::CurZ() {return mCurZ;}

cCIKTabul  cGridIncImageMnt::TheCIK(8,8,-0.5);



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

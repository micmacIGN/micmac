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

#define NoTemplateOperatorVirgule
#define NoSimpleTemplateOperatorVirgule

#include "StdAfx.h"

class cEpipOrientCam : public  ElDistortion22_Gen
{
	friend class PourFairPlaisirAGCC;
	public :
           cEpipOrientCam
	   (
	         bool ToDel,
		 REAL aZoom,
	         CamStenope & aCam,
                 const ElMatrix<REAL> & aRot
           );

           virtual bool OwnInverse(Pt2dr &) const ;    //  Epi vers Phot
           virtual Pt2dr Direct(Pt2dr) const  ;    // Photo vers Epi
           cEpipOrientCam MapingChScale(REAL aChSacle) const;


	private :
	   virtual ~cEpipOrientCam();
           virtual ElDistortion22_Gen  * D22G_ChScale(REAL aS) const; //
           void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  //  Erreur Fatale

	   bool               m2Del;
	   REAL               mZoom;
	   CamStenope *       mCamInit;
	   CamStenopeIdeale   mCamEpip;

};

// Orientation monde->Cam, pour l'entree et la sortie
ElMatrix<REAL>  OrientationEpipolaire(ElRotation3D R1,ElRotation3D R2);

/********************************************************/
/*                                                      */
/*       cEpipOrientCam                                 */
/*                                                      */
/********************************************************/

Pt2dr cEpipOrientCam::Direct(Pt2dr aP) const  
{
   Pt3dr R1,R2;
   mCamInit->F2toRayonR3(aP/mZoom,R1,R2);

   return mCamEpip.R3toF2(R2)*mZoom;
}

bool cEpipOrientCam::OwnInverse(Pt2dr & aP) const     //  Epi vers Phot
{
   Pt3dr R1,R2;
   mCamEpip.F2toRayonR3(aP/mZoom,R1,R2);
   aP  =  mCamInit->R3toF2(R2)*mZoom;
   return true;
}

cEpipOrientCam cEpipOrientCam::MapingChScale(REAL aChSacle) const
{
	return cEpipOrientCam
		(
		    false,
                    mZoom * aChSacle,
		    *mCamInit,
		    mCamEpip.Orient().Mat()
		);
}


cEpipOrientCam::~cEpipOrientCam()
{
   if (m2Del)
      delete mCamInit;
}

ElDistortion22_Gen  * cEpipOrientCam::D22G_ChScale(REAL aS) const
{
    cEpipOrientCam  aNewS = MapingChScale(aS);
    return new cEpipOrientCam(aNewS);
}

void  cEpipOrientCam::Diff(ElMatrix<REAL> & ,Pt2dr) const 
{
      ELISE_ASSERT(false,"No cEpipOrientCam::Diff");
}

cEpipOrientCam::cEpipOrientCam
(
	         bool ToDel,
		 REAL aZoom,
	         CamStenope & aCam,
                 const ElMatrix<REAL> & aMat
) :
  m2Del    (ToDel),
  mZoom    (aZoom),
  mCamInit (&aCam),
  mCamEpip (true,aCam.Focale(),aCam.PP(),aCam.ParamAF())
{
   ELISE_ASSERT(!aCam.UseAFocal(),"AFocal epip");
   ElRotation3D aRot(Pt3dr(0,0,0),aMat);
   ElMatrix<REAL> MR = aRot.Mat();

   Pt3dr Tr = -(MR*aCam.VraiOpticalCenter());

   mCamEpip.SetOrientation(ElRotation3D(Tr,MR));
}

/********************************************************/
/*                                                      */
/*                ::                                    */
/*                                                      */
/********************************************************/
ElMatrix<REAL>  OrientationEpipolaire(ElRotation3D R1,ElRotation3D R2)
{
     Pt3dr COpt1 = R1.ImRecAff(Pt3dr(0,0,0));
     Pt3dr COpt2 = R2.ImRecAff(Pt3dr(0,0,0));

     Pt3dr Ox = vunit(COpt2 - COpt1);

     Pt3dr OX12 = vunit(R1.IRecVect(Pt3dr(1,0,0)) +R2.IRecVect(Pt3dr(1,0,0)));
     if (scal(Ox,OX12) <0)
         Ox =  -Ox;

     Pt3dr OZ1 = R1.IRecVect(Pt3dr(0,0,1));
     Pt3dr OZ2 = R2.IRecVect(Pt3dr(0,0,1));
     Pt3dr Oz = vunit(OZ2+OZ1) ;

     Oz = vunit(Oz - Ox * scal(Ox,Oz));

     Pt3dr Oy = Oz ^ Ox;

     ElMatrix<REAL> aRes(3,3);

     SetCol(aRes,0,Ox);
     SetCol(aRes,1,Oy);
     SetCol(aRes,2,Oz);

     return aRes.transpose();
}

/********************************************************/
/*                                                      */
/*                CpleEpipolaireCoord                   */
/*                                                      */
/********************************************************/


CpleEpipolaireCoord * CpleEpipolaireCoord::CamEpipolaire
                      (
                          CamStenope  & aCam1, Pt2dr aP1,
                          CamStenope  & aCam2, Pt2dr aP2,
                          REAL aZoom
                      )
{
   aP1 = aP1 * aZoom;
   aP2 = aP2 * aZoom;
   ElMatrix<REAL> aMat =  OrientationEpipolaire(aCam1.Orient(),aCam2.Orient());

   cEpipOrientCam * OR1 = new cEpipOrientCam(false,aZoom,aCam1,aMat);
   cEpipOrientCam * OR2 = new cEpipOrientCam(false,aZoom,aCam2,aMat);

    cMappingEpipCoord * aMap1 = new cMappingEpipCoord(OR1,true);
    cMappingEpipCoord * aMap2 = new cMappingEpipCoord(OR2,true);


    Pt2dr aQ1 = aMap1->Direct(aP1);
    Pt2dr aQ2 = aMap2->Direct(aP2);

    aMap1->AddTrFinale(Pt2dr(aQ2.x-aQ1.x,0));
    return new CpleEpipolaireCoord(aMap1,aMap2);
}


/********************************************************/
/*                                                      */
/*                cCpleEpip                             */
/*                                                      */
/********************************************************/


CamStenopeIdeale  cCpleEpip::CamOut(const CamStenope & aCamIn,Pt2dr aPP,Pt2di aSz)
{
    ElRotation3D  aRC2M(aCamIn.VraiOpticalCenter(),mMatC2M);
    CamStenopeIdeale  aCamOut(aCamIn.DistIsC2M(),mFoc,aPP,aCamIn.ParamAF());
    aCamOut.SetOrientation(aRC2M.inv());
    aCamOut.SetSz(aSz);

    return aCamOut;
}

Box2dr  cCpleEpip::BoxCam(const CamStenope & aCamIn,const CamStenope & aCamOut) const
{
    Box2dr aBoxIn (Pt2dr(0,0),Pt2dr(aCamIn.Sz()));
    std::vector<Pt2dr> aVPtsIn;
    aBoxIn.PtsDisc(aVPtsIn,6);

    Pt2dr aPInfOut(1e20,1e20);
    Pt2dr aPSupOut(-1e20,-1e20);

    for (int aK=0 ; aK<int(aVPtsIn.size()) ; aK++)
    {
        Pt2dr aP = TransfoEpip(aVPtsIn[aK],aCamIn,aCamOut);
        aPInfOut.SetInf(aP);
        aPSupOut.SetSup(aP);
    }

    return Box2dr(aPInfOut,aPSupOut);
}




Pt2dr cCpleEpip::TransfoEpip
      (
            const Pt2dr & aPIm,
            const CamStenope & aCamIn,
            const CamStenope & aCamOut
       ) const
{

    Pt3dr aC = aCamIn.VraiOpticalCenter();
    Pt3dr aRay = aCamIn.F2toDirRayonR3(aPIm);
    return  aCamOut.R3toF2(aC+aRay);
}



cCpleEpip::cCpleEpip
(
   double aScale,
   const CamStenope & aC1,
   const CamStenope & aC2
)  :
   mScale    (aScale),
   mCInit1   (aC1),
   mCInit2   (aC2),
   mSzIn     (Sup(mCInit1.Sz(),mCInit2.Sz())),
   mFoc      (sqrt(aC1.Focale()*aC2.Focale())/mScale),
   mMatM2C   (OrientationEpipolaire(mCInit1.Orient(),mCInit2.Orient())),
   mMatC2M   (mMatM2C.transpose()),
   mCamOut1  (CamOut(mCInit1,Pt2dr(0,0),mSzIn)),
   mCamOut2  (CamOut(mCInit2,Pt2dr(0,0),mSzIn))
{
   Box2dr aB1 = BoxCam(mCInit1,mCamOut1);
   Box2dr aB2 = BoxCam(mCInit2,mCamOut2);

   Box2dr aB = Sup(aB1,aB2);

   mCamOut1  =CamOut(mCInit1,-aB._p0,Pt2di(aB.sz()));
   mCamOut2  =CamOut(mCInit2,-aB._p0,Pt2di(aB.sz()));

}


void cCpleEpip::ImEpip(Tiff_Im aTIn,bool Im1,const std::string & aNameImOut)
{
    const CamStenope & aCamIn =        Im1 ? mCInit1  : mCInit2;
    const CamStenopeIdeale & aCamOut = Im1 ? mCamOut1 : mCamOut2;
    Pt2di aSzOut = aCamOut.Sz();

    std::vector<Im2DGen *>   aVIn = aTIn.VecOfIm(aCamIn.Sz());
    ELISE_COPY(aTIn.all_pts(),aTIn.in(),StdOutput(aVIn));
    std::vector<Im2DGen *>   aVOut = aTIn.VecOfIm(aSzOut);

    cKernelInterpol1D * aKern = cKernelInterpol1D::StdInterpCHC(mScale,100);
    int aNbCh = aVIn.size();

    for (int anX=0; anX<aSzOut.x ; anX++)
    {
       for (int anY=0; anY<aSzOut.y ; anY++)
       {
            Pt2dr aPIm = TransfoEpip(Pt2dr(anX,anY),aCamOut,aCamIn);
            for (int aK=0 ; aK<aNbCh ; aK++)
            {
                double aVal = aKern->Interpole(*(aVIn[aK]),aPIm.x,aPIm.y);
                aVOut[aK]->TronqueAndSet(Pt2di(anX,anY),aVal);
            }
       }
    }
    delete aKern;

    GenIm::type_el aTypeNOut =  aTIn.type_el();
    Tiff_Im aTOut
            (
                  aNameImOut.c_str(),
                  aSzOut,
                  aTypeNOut,
                  Tiff_Im::No_Compr,
                  aTIn.phot_interp()
            );

   ELISE_COPY
   (
        aTOut.all_pts(),
        StdInPut(aVOut),
        aTOut.out()
   );



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

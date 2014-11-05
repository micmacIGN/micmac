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
#include "hough_include.h"


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

INT ElHoughImplem::NbRabTeta() const {return mNbRabTeta;}
INT ElHoughImplem::NbRabRho()  const {return mNbRabRho;}

Im2D_INT4 ElHoughImplem::Pds(Im2D_U_INT1 im)
{
   Transform(mHouhAccul,im);
   return mHouhAccul;
}

Im2D_INT4 ElHoughImplem::PdsAng
          (
               Im2D_U_INT1 ImRho,
               Im2D_U_INT1 ImTeta,
               REAL        Incert,
               bool        IsGrad
          ) 
{
    Transform_Ang(mHouhAccul,ImRho,ImTeta,Incert,IsGrad);
    return mHouhAccul;
}

Im2D_INT4 ElHoughImplem::PdsInit()
{
   return mHouhAcculInit;
}






ElHoughImplem::ElHoughImplem
(
     Pt2di SzXY,
     REAL  StepRho ,
     REAL  StepTeta ,
     REAL  RabRho,
     REAL  RabTeta
) :
   ElHough(SzXY,StepTeta),
   mCX          (NbX()/2.0),
   mCY          (NbY()/2.0),
   mRabRho      (RabRho),
   mNbRabRho    (round_up(RabRho/StepRho)),
   mStepTeta    (PI/NbTeta()),
   mNbRabTeta   (round_ni(RabTeta/mStepTeta)),
   mNbTetaTot   (2*mNbRabTeta+NbTeta()),
   mRabTeta     (mStepTeta * mNbRabTeta),
   mStepRhoInit (StepRho),
   mStepRho     (NbTetaTot(),StepRho),
   mDataSRho    (mStepRho.data()),
   mIRhoMin     (0),
   mIRhoMax     (0),
   mNbCelTot    (0),
   mAdrElem     (NbX(),NbY(),0),
   mDataAdE     (mAdrElem.data()),
   mNbElem      (NbX(),NbY(),0),
   mDataNbE     (mNbElem.data()),
   mIndRho      (1),
   mDataIRho    (0),
   mIndTeta     (1),
   mDataITeta   (0),
   mGetTetaTrivial (1),
   mPds         (1),
   mDataPds     (0),
   mHouhAccul   (1,1),
   mDataHA      (0),
   mHouhAcculInit (1,1),
   mDataHAInit  (0),
   mImageXY     (1,1),
   mDataImXY    (0),
   mImageRT     (1,1),
   mDataImRT    (0),
   mPdsEcTeta   (NbTeta()),
   mDataPdsEcTeta (mPdsEcTeta.data()),
   mMarqBCVS      (1,1),
   mDMV_IsCalc    (false)
{
	      cout << "Rajouter Check-Sum memoire dans Hough " << StepTeta << "\n";
}


void ElHoughImplem::PostInit()
{
   ELISE_ASSERT(NbTeta()< mIndTeta.vmax(),"NbTeta out Limits in hough");
}


ElHoughImplem * ElHoughImplem::NewOne
(
     Pt2di SzXY,
     REAL  StepRho ,
     REAL  StepTeta,
     Mode  mode,
     REAL  RabRho, 
     REAL  RabTeta 
)
{
   ElHoughImplem * EHI = 0;
   switch (mode)
   {
         case ModePixelExact :
              EHI = SubPixellaire(SzXY,StepRho,StepTeta,RabRho,RabTeta);
         break;

         case ModeBasic :
              EHI = Basic(SzXY,StepRho,StepTeta,false,false,RabRho,RabTeta);
         break;

         case ModeStepAdapt :
              EHI = Basic(SzXY,StepRho,StepTeta,true,false,RabRho,RabTeta);
         break;

         case ModeBasicSubPix :
              EHI = Basic(SzXY,StepRho,StepTeta,false,true,RabRho,RabTeta);
         break;

   }
   EHI->finish();
   EHI->clean();
   return EHI;
}


INT ElHoughImplem::GetIndTeta(INT AValue,tElIndex *tab,INT Nb)
{
    if (mGetTetaTrivial)
       return AValue;

    return (INT) (ElSTDNS lower_bound(tab,tab+Nb,AValue)-tab);
}


static bool CmpPairOnTeta
            (
               const std::pair<Pt2di,REAL> &p1,
               const std::pair<Pt2di,REAL> &p2
            )
{
    return p1.first.y < p2.first.y;
}

void ElHoughImplem::finish()
{
   ElSTDNS vector<ElSTDNS pair<Pt2di,REAL> >  tmp;

   mNbCelTot = 0;
   REAL  PdsCelMax =0;

   for (INT x=0; x<NbX() ; x++)
   {
       for (INT y=0; y<NbY() ; y++)
       {
            tmp.clear();
            ElemPixel(tmp,Pt2di(x,y));
            for
            (
                tLCel::const_iterator it=tmp.begin();
                it!= tmp.end();
                it++
            )
            {
                 mNbCelTot++;
                 ElSetMax(PdsCelMax,it->second);
                 ElSetMin(mIRhoMin,it->first.x);
                 ElSetMax(mIRhoMax,it->first.x);
            }
       }
   }

   {
       INT AmplRho = ElMax(ElAbs(mIRhoMax+mNbRabRho),ElAbs(mIRhoMin-mNbRabRho));
       mIRhoMax = + AmplRho;
       mIRhoMin = - AmplRho;
   }

   SetNbRho(mIRhoMax-mIRhoMin+1);
   ELISE_ASSERT(NbRho()<mIndRho.vmax(),"NbRho should be < 256 in hough");

   mIndRho = tImIndex(mNbCelTot,0);
   mDataIRho = mIndRho.data();
   mIndTeta = tImIndex(mNbCelTot,0);
   mDataITeta = mIndTeta.data();
   mPds = Im1D_U_INT1(mNbCelTot,0);
   mDataPds = mPds.data();

   mFactPds = 255/PdsCelMax;

   mHouhAccul = Im2D_INT4(mNbTetaTot,NbRho(),0);
   mDataHA = mHouhAccul.data();
   mHouhAcculInit = Im2D_INT4(mNbTetaTot,NbRho(),0);
   mDataHAInit = mHouhAcculInit.data();

   mMarqBCVS    =  Im2D_U_INT1 (mNbTetaTot,NbRho(),0);

   mNbCelTot = 0;
   {
   for (INT x=0; x<NbX() ; x++)
   {
       for (INT y=0; y<NbY() ; y++)
       {
            tmp.clear();
            ElemPixel(tmp,Pt2di(x,y));
            mDataAdE[y][x] = mNbCelTot;

            std::sort(tmp.begin(),tmp.end(),CmpPairOnTeta);
            
            for
            (
                tLCel::const_iterator it=tmp.begin();
                it!= tmp.end();
                it++
            )
            {
                 mDataIRho[mNbCelTot] = it->first.x - mIRhoMin;
                 mDataITeta[mNbCelTot] = it->first.y;
                 mDataPds[mNbCelTot] = round_ni(it->second * mFactPds);

                 if (it == tmp.begin())
                 {
                     if (it->first.y !=0)
                     {
                        mGetTetaTrivial = 0;
                     }
                 }
                 else
                 {
                     if (it->first.y !=  mDataITeta[mNbCelTot-1]+1)
                     {
                        mGetTetaTrivial = 0;
                     }
                 }

                 mNbCelTot++;
            }
            mDataNbE[y][x] = mNbCelTot-mDataAdE[y][x];
       }
   }
   }

   Im2D_U_INT1 ImUnif(NbX(),NbY(),1);
   Transform(mHouhAcculInit,ImUnif);
}

Seg2d ElHoughImplem::Grid_Hough2Euclid(Pt2dr p) const 
{
   INT iTeta = round_ni(p.x);
   REAL  aRho = PosG2S_Rho(p.y,iTeta);
   REAL  aTeta =  G2S_Teta(p.x);
   
   Pt2dr aDir = Pt2dr::FromPolar(1,aTeta); 
   Pt2dr p1   = aDir * aRho;
   Pt2dr tgt   =  aDir * Pt2dr(0,1000);
   
   Seg2d res(S2G_XY(p1-tgt),S2G_XY(p1+tgt));


   return res.clip(Box2di(Pt2di(-1,-1),Pt2di(NbX()+1,NbY()+1)));
}



Pt2dr ElHoughImplem::Grid_Euclid2Hough(Seg2d seg) const 
{
     {
            Pt2dr v01 = seg.v01();
            if ((v01.x >0) || ((v01.x==0) &&(v01.y<0)))
               seg = seg.reverse();
     }
    SegComp GSeg(G2S_XY(seg.p0()),G2S_XY(seg.p1()));

    REAL aSRho = GSeg.ordonnee(Pt2dr(0,0));
    REAL aSTeta = angle(GSeg.tangente()/Pt2dr(0.0,1.0));

    REAL aGTeta = S2G_Teta(aSTeta);
    REAL aGRho = PosS2G_Rho(aSRho,round_ni(aGTeta));

    return Pt2dr(aGTeta,aGRho);
}


/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

ElHough::~ElHough() {}

REAL ElHough::LongEstimTeta() const
{
   return ElMax(mNbX,mNbY)/2.0;
}

ElHough::ElHough(Pt2di SzXY,REAL StepTeta) :
   mStepTetaInit (StepTeta),
   mNbX    (SzXY.x),
   mNbY    (SzXY.y),
   mNbTeta (round_ni(PI / ( StepTeta / LongEstimTeta())))
   
{
}

void ElHough::SetNbRho(INT aNb)
{
   mNbRho = aNb;
}

ElHough * ElHough::NewOne
          (
             Pt2di Sz,
             REAL StepRho,
             REAL StepTeta,
             Mode mode,
             REAL RabRho,
             REAL RabTeta
          )
{
   ElHough * res = ElHoughImplem::NewOne
                   (
                      Sz,
                      StepRho,
                      StepTeta,
                      mode,
                      ElMax(0.0,RabRho),
                      ElMax(0.0,RabTeta)
                   );

   res->mMode = mode;
   return res;
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

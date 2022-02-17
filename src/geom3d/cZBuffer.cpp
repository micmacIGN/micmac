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


static const REAL Eps = 1e-7;

static const int SzDalleDef = 50;


// static bool AssertT0 = false;

cZBuffer::cZBuffer
(
    Pt2dr OrigineIn,
    Pt2dr StepIn,
    Pt2dr OrigineOut,
    Pt2dr StepOut
) :
   mOrigineIn  (OrigineIn),
   mStepIn     (StepIn),
   mOrigineOut (OrigineOut),
   mStepOut    (StepOut),
   mRes        (1,1),
   mDataRes    (0),
   mImOkTer    (1,1),
   mTImOkTer   (mImOkTer),
   mImTriInv   (1,1),
   mDynEtire   (-1),
   mImEtirement(1,1),
   mWihBuf     (true),
   mBufDone    (false),
   mImX3       (1,1),
   mTX3        (mImX3),
   mImY3       (1,1),
   mTY3        (mImY3),
   mImZ3       (1,1),
   mTZ3        (mImZ3),
   mEpsIntInv  (1e-5),
   mTImDef_00  (Pt2di(1,1)),
   mTImDef_10  (Pt2di(1,1)),
   mTImDef_01  (Pt2di(1,1)),
   mTImDef_11  (Pt2di(1,1)),
   mIsRPC      (false)
{
}

void cZBuffer::SetRPC(bool IsRPC,double aZMin,double aZMax)
{
   mIsRPC = IsRPC;
   mZMinRPC = aZMin;
   mZMaxRPC = aZMax;
}


bool cZBuffer::RPCIsBascVisible(const Pt3dr & aP) const
{
   return true;
}


void cZBuffer::SetEpsilonInterpoleInverse(double anEps)
{
   mEpsIntInv = anEps;
}

void  cZBuffer::InitDynEtirement(double aDyn)
{
    mDynEtire = aDyn;
    ELISE_ASSERT(!mBufDone,"Internal Error in cZBuffer; Cannot Set Etyr since Buf Done");
    if (aDyn >0)
        mWihBuf = true;
}


Im2D_U_INT1 cZBuffer::ImEtirement()
{
   ELISE_ASSERT(mImEtirement.sz().x>1,"cZBuffer::ImEtirement Non Init");
   return mImEtirement;
}

void cZBuffer::SetWithBufXYZ(bool aWBuf)
{
    mWihBuf  = aWBuf;
}

cZBuffer::~cZBuffer() {}
double cZBuffer::ZInterpofXY(const Pt2dr & aP,bool & OK) const
{
   ELISE_ASSERT(false,"cZBuffer::ZInterpofXY");
   return 0;
}
Pt3dr  cZBuffer::InvProjTerrain(const Pt3dr &) const
{
   ELISE_ASSERT(false,"cZBuffer::ProjInverse");
   return Pt3dr(0,0,0);
}


Pt2di cZBuffer::P0_Out() const 
{
   return mOffet_Out_00;
}

Pt2di cZBuffer::SzOut() const 
{
    return mSzRes;
}


Pt2di cZBuffer::ToPtIndexDef(const Pt2di & aPt) const
{
   return Pt2di((aPt.x-mP0In.x)/SzDalleDef,(aPt.y-mP0In.y)/SzDalleDef);
}

bool DEBUG_ZBB = false;
Pt2di PT_BugZBB;
bool ERupnik_MM();

static Pt2di ORIG_BoxBug(735,484);


Im2D_REAL4 cZBuffer::Basculer
           (
               Pt2di & aOffset_Out_00,
               Pt2di aP0In,
               Pt2di aP1In,
               float aDef,
               bool * isOk
           )
{
/*
{
    std::cout << "cZBuffer::Basculer \n";
    getchar();

}
*/

    if (isOk) 
    {
        *isOk = true;
    }
    mBufDone = false;
    mP0In = aP0In;
    mSzIn =  aP1In-aP0In;

    if (mWihBuf)
    {
       mImX3 = Im2D<tElZB,REAL8>(mSzIn.x,mSzIn.y,22);
       mDX3  = mImX3.data();
       mTX3 = TIm2D<tElZB,REAL8>(mImX3);
       mImY3 = Im2D<tElZB,REAL8>(mSzIn.x,mSzIn.y,23);
       mDY3  = mImY3.data();
       mTY3 = TIm2D<tElZB,REAL8>(mImY3);
       mImZ3 = Im2D<tElZB,REAL8>(mSzIn.x,mSzIn.y,240);
       mDZ3  = mImZ3.data();
       mTZ3 = TIm2D<tElZB,REAL8>(mImZ3);
    }
    mOffet_Out_00 = Pt2di(0,0);
    mImOkTer = Im2D_Bits<1>(mSzIn.x,mSzIn.y,0);
    mTImOkTer = TIm2DBits<1>(mImOkTer);

    Pt2dr aPInf(1e30,1e30),aPSup(-1e30,-1e30);
    Pt2di aPIn;
    
    int aNbPts=0,aNbOkTer=0,aNbOkIm=0;

    int aSzXDef = (aP1In.x-aP0In.x + SzDalleDef -1) / SzDalleDef;
    int aSzYDef = (aP1In.y-aP0In.y + SzDalleDef -1) / SzDalleDef;
    Pt2di aSzDef(aSzXDef,aSzYDef);
    mTImDef_00.Resize(aSzDef);
    mTImDef_10.Resize(aSzDef);
    mTImDef_01.Resize(aSzDef);
    mTImDef_11.Resize(aSzDef);

static int aCpt=0; aCpt++;

    // fait pour verifier par cher que le gradient est raisonnable
    
    Pt2di aLastPIn = aP0In - Pt2di(100000,100);
    Pt3dr aLastP3Out(0,0,0);
    double aLastZofXY=1e20;
    double aMaxGrad = 0.0;
    double aNbBadGrad = 0.0;
    double aNbGrad = 0.0;

double aDebugMaxGr =0 ;
Pt3dr aPtOutMaxGrad(0,0,0);
Pt2di aPtInMaxGrad(0,0);
Pt3dr aLastPtOutMaxGrad(0,0,0);
Pt2di aLastPtInMaxGrad(0,0);

    for (int anXDal0 = aP0In.x ;  anXDal0< aP1In.x ; anXDal0 += SzDalleDef  )
    {
        int anXDal1 = ElMin(aP1In.x,anXDal0+SzDalleDef);
        for (int anYDal0 = aP0In.y ;  anYDal0< aP1In.y ; anYDal0 += SzDalleDef  )
        {
             int anYDal1 = ElMin(aP1In.y,anYDal0+SzDalleDef);
             std::vector<double> aVZofXY;
             for (aPIn.x=anXDal0 ; aPIn.x<anXDal1; aPIn.x++)
             {
                  for (aPIn.y=anYDal0 ; aPIn.y<anYDal1; aPIn.y++)
                  {

bool TEST = (aPIn.x==2000) && (aPIn.y==2000) && MPD_MM();
			aNbPts++;
			if (SelectP(aPIn))
			{
				aNbOkTer++;
                                double aZofXY;

//static int aCpt=0; aCpt++; if (aCpt>10600000) std::cout << "Iiiinnnn " << aCpt << "\n";
				Pt3dr aP3Out = ProjDisc(aPIn,&aZofXY);
if (TEST)
{
    std::cout << "P3OUT " <<  aP3Out << "\n";
   Pt3dr aPInTer =  ToCoordInAbs(Pt3dr(aPIn.x,aPIn.y,ZofXY(aPIn)));
   Pt3dr aPOutTer = ProjTerrain(aPInTer);
   Pt3dr  aPFIN = ToCoordOutLoc(aPOutTer);

    std::cout << "TTTT " << aPIn << "\n";
    std::cout << "TTTT " << aPInTer << "\n";
    std::cout << "TTTT " << aPOutTer << "\n";
    std::cout << "TTTT " << aPFIN << "\n";

}
//if (aCpt>10600000)std::cout << "OooUuutt " << aCpt << "\n";
				Pt2dr aP2Out(aP3Out.x,aP3Out.y);

                                double aDist2 =  square_euclid(aPIn-aLastPIn);
                                if ( aDist2 < 10)
                                {
                                    double aDist3 = sqrt(aDist2+ElSquare(aZofXY-aLastZofXY));
                                    double aGrad = euclid(aP2Out-Pt2dr(aLastP3Out.x,aLastP3Out.y)) / aDist3;
                                    aNbGrad ++;
                                    if (aGrad>1.5)
                                    {
                                        if (0 && ((MPD_MM() || ERupnik_MM()) && ((aZofXY>40) && (aZofXY<250))))
                                        {
                                            // std::cout << "ZZZZZZzBadGraddd  " << aZofXY  << " " << aP3Out << "\n";
                                            if (aGrad>aDebugMaxGr)
                                            {
                                               aDebugMaxGr = aGrad;
                                               aPtOutMaxGrad  = aP3Out;
                                               aPtInMaxGrad = aPIn;

                                               aLastPtOutMaxGrad  = aLastP3Out;
                                               aLastPtInMaxGrad = aLastPIn;
                                            }
                                        }
                                        aNbBadGrad++;
                                    }
                                    if (aGrad > aMaxGrad)
                                    {

if (0 && MPD_MM())
{
   std::cout.precision(12);
   std::cout << "GRAD " << aGrad  << " D2="<< aDist2 << " D3=" << aDist3  << "\n";
   std::cout << " IN=" <<  aPIn << aLastPIn <<  aPIn- aLastPIn<< "\n";
   double  aValZofXY = ZofXY(aPIn);
   double  aLastValZofXY = ZofXY(aLastPIn);
   Pt3dr anAbsIn = ToCoordInAbs(Pt3dr(aPIn.x,aPIn.y,aValZofXY));
   Pt3dr aPLastIn = ToCoordInAbs(Pt3dr(aLastPIn.x,aLastPIn.y,aLastValZofXY));

std::cout << "RPCCVISSSSSSSSSSSSSSSss " << RPCIsBascVisible(anAbsIn) <<  " " << RPCIsBascVisible(aPLastIn)  << "\n";

   std::cout << " IN-ABS=" <<  anAbsIn << aPLastIn <<   "\n";
   std::cout << " OUT=" <<  aP3Out << aLastP3Out << aP3Out-aLastP3Out << "\n";
   std::cout << "\n";
   // getchar();
}
                                        aMaxGrad = aGrad;
                                    }
                                }

if (TEST)
{
   std::cout << "OOOuut " << aP2Out << "\n";
}


				if (SelectPBascul(aP2Out))
				{
				   aNbOkIm++;


if (0 && (aMaxGrad>400)  &&(MPD_MM() || ERupnik_MM()))
{


//  std::cout << "pppppppppppppp " << aP2Out << "\n";
// if (euclid(aP2Out) > 1e5  )
{
   std::cout << "STEPIN " << mStepIn << " OOOO " << aZofXY << "\n";
   DEBUG_ZBB = true;
   for (int aDx = -1 ; aDx<=1 ; aDx++)
   {
       for (int aDy = -1 ; aDy<=1 ; aDy++)
       {
           PT_BugZBB =  Pt2di(aDx,aDy);
           Pt2di aPV = aPIn + PT_BugZBB;
           std::cout << "uuu DEBUG_ZBB " << aPV << " " << aZofXY << "\n";
           // mGeom.Objet2ImageInit_Euclid(Pt2dr(aPV),&aZofXY);
           std::cout << " ZzzZzz " <<  ProjDisc(aPV,&aZofXY) << " " << aZofXY << "\n";
           //  ProjDisc(aPV,&aZofXY);

       }
   }

   std::cout << "----aP2OutaP2Out " << aP2Out  << " " << aZofXY  << " " << aPIn << " BD=" <<  mBufDone << "\n";
   ProjDisc(aPIn,&aZofXY);
   DEBUG_ZBB = false;
   std::cout << "Donnee aP2OutaP2Out\n";
   exit(-1);
}
}
				   aPInf.SetInf(aP2Out);
				   aPSup.SetSup(aP2Out);

				   if (mWihBuf)
				   {
					  mDX3[aPIn.y-aP0In.y][aPIn.x-aP0In.x] = (tElZB)aP3Out.x;
					  mDY3[aPIn.y-aP0In.y][aPIn.x-aP0In.x] = (tElZB)aP3Out.y;
					  mDZ3[aPIn.y-aP0In.y][aPIn.x-aP0In.x] = (tElZB)aP3Out.z;
				   }
				   mImOkTer.set(aPIn.x-mP0In.x,aPIn.y-mP0In.y,1);
                                   if (mDynEtire >0) aVZofXY.push_back(aZofXY);
				}
                                aLastPIn = aPIn;
                                aLastP3Out = aP3Out;
                                aLastZofXY = aZofXY;
			}
                }
            }

            if (mDynEtire >0)
            {
                // Pt2di anIndexDef((anXDal0-aP0In.x)/SzDalleDef,(anYDal0-aP0In.y)/SzDalleDef);
                Pt2di anIndexDef = ToPtIndexDef(Pt2di(anXDal0,anYDal0));
                int aNbVal = (int)aVZofXY.size();

                if (aNbVal)
                {
                      double aZMed  = KthVal(VData(aVZofXY),aNbVal,aNbVal/2);
                      Pt3dr aPMed((anXDal0+anXDal1)/2.0,(anYDal0+anYDal1)/2.0,aZMed);

                      Pt3dr aDerX = (ProjDisc(aPMed+Pt3dr(1,0,0)) - ProjDisc(aPMed+Pt3dr(-1,0,0))) / 2.0;
                      Pt3dr aDerY = (ProjDisc(aPMed+Pt3dr(0,1,0)) - ProjDisc(aPMed+Pt3dr(0,-1,0))) / 2.0;

                      ElMatrix<double> aJac =  MatFromCol(Pt2dr(aDerX.x,aDerX.y),Pt2dr(aDerY.x,aDerY.y));

                      aJac = gaussj(aJac);
                      mTImDef_00.oset(anIndexDef,aJac(0,0));
                      mTImDef_10.oset(anIndexDef,aJac(1,0));
                      mTImDef_01.oset(anIndexDef,aJac(0,1));
                      mTImDef_11.oset(anIndexDef,aJac(1,1));
                }
                else
                {
                   mTImDef_00.oset(anIndexDef,0.0);
                   mTImDef_10.oset(anIndexDef,0.0);
                   mTImDef_01.oset(anIndexDef,0.0);
                   mTImDef_11.oset(anIndexDef,0.0);
                }
            }
        }
    } 



    if (mIsRPC && (aMaxGrad > 100))
    {
        std::cout << "GRAD=" << aMaxGrad << " PropBadGrad " << aNbBadGrad / aNbGrad << "\n";
        ELISE_ASSERT(false,"probable instability in RPCi due to bad heigth interval");
    }
    aOffset_Out_00 = mOffet_Out_00 = round_down(aPInf);
    if (mWihBuf)
    {
       mBufDone = true;
       for (aPIn.x=aP0In.x ; aPIn.x<aP1In.x; aPIn.x++)
       {
           for (aPIn.y=aP0In.y ; aPIn.y<aP1In.y; aPIn.y++)
           {
              mDX3[aPIn.y-aP0In.y][aPIn.x-aP0In.x] -= mOffet_Out_00.x;
              mDY3[aPIn.y-aP0In.y][aPIn.x-aP0In.x] -= mOffet_Out_00.y;
           }
       }
    }


    mSzRes = round_up(aPSup) - mOffet_Out_00;

    if ((mSzRes.x<=1)  || (mSzRes.y<=1))
    {
       if (isOk) 
       {
           *isOk = false;
       }
       else
       {
           ELISE_ASSERT(false,"Cannot Bascule Nuage");
       }
       return  Im2D_REAL4(1,1,aDef);
    }



    mRes = Im2D_REAL4(mSzRes.x,mSzRes.y,aDef);
    mImTriInv = Im2D_Bits<1>(mSzRes.x,mSzRes.y,0);
    mDataRes = mRes.data();

    mImAttrOut.clear();
    for (int aKA=0 ; aKA <int(mImAttrIn.size()) ; aKA++)
    {
        mImAttrOut.push_back(mImAttrIn[aKA]->ImOfSameType(mSzRes));
    }

    if (mDynEtire > 0)
       mImEtirement = Im2D_U_INT1(mSzRes.x,mSzRes.y,0);

    for (int x=aP0In.x ; x<aP1In.x-1; x++)
    {
        for (int y=aP0In.y ; y<aP1In.y-1; y++)
        {
                Pt2di P00(x,y);
                Pt2di P10(x+1,y);
                Pt2di P01(x,y+1);
                Pt2di P11(x+1,y+1);

                BasculerUnTriangle(P00,P10,P11,true);
                BasculerUnTriangle(P00,P11,P01,false);
        }
    }

    return mRes;
}

Im2D_REAL4 cZBuffer::ZCaches
           (
               Im2D_REAL4 aMnt,
               Pt2di aOffset_Out_00,
               Pt2di aP0In,
               Pt2di aP1In,
               float aDef
           )
{
    mOffet_Out_00 = aOffset_Out_00;
    TIm2D<REAL4,REAL> aTMnt(aMnt);

    Im2D_REAL4 aRes(aP1In.x-aP0In.x,aP1In.y-aP0In.y,0.0);
    TIm2D<REAL4,REAL> aTRes(aRes);

    Pt2di aPIn;
    for (aPIn.x=aP0In.x ; aPIn.x<aP1In.x; aPIn.x++)
    {
        for (aPIn.y=aP0In.y ; aPIn.y<aP1In.y; aPIn.y++)
        {
           aTRes.oset(aPIn-aP0In,0.0);
	   if (SelectP(aPIn))
	   {

	       Pt3dr aP3Out = ProjDisc(aPIn);
               Pt2dr aP2ROut(aP3Out.x,aP3Out.y);
               Pt2di aP2IOut = round_down(aP2ROut);
               bool OK =    (aTMnt.getproj(aP2IOut)!=aDef)
                         && (aTMnt.getproj(aP2IOut+Pt2di(1,0))!=aDef)
                         && (aTMnt.getproj(aP2IOut+Pt2di(0,1))!=aDef)
                         && (aTMnt.getproj(aP2IOut+Pt2di(1,1))!=aDef);
               if (OK)
               {
                   double aZProj = aTMnt.getprojR(aP2ROut);
                   aTRes.oset(aPIn-aP0In,ElMax(0.0,aZProj-aP3Out.z));
               }
	   }
        }
    }
    return aRes;
}


Im2D_REAL4 cZBuffer::ZCaches ( Pt2di aP0In, Pt2di aP1In, float aDef)
{
   Pt2di aOffset_Out_00;
   Im2D_REAL4 aMnt = Basculer(aOffset_Out_00,aP0In,aP1In,aDef);

   return ZCaches(aMnt,aOffset_Out_00,aP0In,aP1In,aDef);
}


bool cZBuffer::SelectP(const Pt2di & aP)   const 
{
   return true;
}

bool cZBuffer::SelectPBascul(const Pt2dr & aP)   const 
{
   return true;
}

Im2D_Bits<1> cZBuffer::ImOkTer() const
{
   return mImOkTer;
}
Im2D_Bits<1> cZBuffer::ImTriInv() const
{
   return mImTriInv;
}




bool  cZBuffer::OkTer(const Pt2di & aP) const
{
   return (mImOkTer.get_def(aP.x,aP.y,0)!=0);
}




std::vector<Im2DGen *> cZBuffer::AttrOut()
{
   return mImAttrOut;
}

void cZBuffer::AddImAttr(Im2DGen * anIm)
{
   return mImAttrIn.push_back(anIm);
}



// WU =>  The 3 point are point of a MNS here, but is we be used as any triangle
void cZBuffer::BasculerUnTriangle(Pt2di A,Pt2di B,Pt2di C,bool TriBas)
{

   if (
           (! mTImOkTer.get(A-mP0In))
        || (! mTImOkTer.get(B-mP0In))
        || (! mTImOkTer.get(C-mP0In))
      )
      return;
  
    // WU=> Here the proj disc  method go from ground to image
    // at this step for each point x,y must be pixel coordinates, and z the depth
     Pt3dr A3  =  ProjDisc(A);
     Pt3dr B3  =  ProjDisc(B);
     Pt3dr C3  =  ProjDisc(C);


     Pt2dr A2(A3.x,A3.y);
     Pt2dr B2(B3.x,B3.y);
     Pt2dr C2(C3.x,C3.y);

     Pt2dr AB = B2-A2;
     Pt2dr AC = C2-A2;
     REAL aDet = AB^AC;  //WU :  computue determinant to see if triangle is well oriented


	 //Calcul de l'etirement du triangle
     int aCoefEtire= -1;
     double aCoefEtirReel=-1;
     if (mDynEtire>0)   // WU this part is not important, it compute some streching caracteristic : goto ENDSKIP
     {
        Pt2dr u = TriBas ? (B2-A2) : (C2-B2);
        Pt2dr v = TriBas ? (C2-B2) : (C2-A2);

        // On tient compte du jacobien inverse pour avoir une estimation plus intrinseque de l'etirement
        Pt2di anIndexDef = ToPtIndexDef(A);
        double m00 = mTImDef_00.get(anIndexDef);
        double m10 = mTImDef_10.get(anIndexDef);
        double m01 = mTImDef_01.get(anIndexDef);
        double m11 = mTImDef_11.get(anIndexDef);

        Pt2dr aU (u.x*m00 + v.x*m01 , u.y*m00 +v.y *m01);
        Pt2dr aV (u.x*m10 + v.x*m11 , u.y*m10 +v.y *m11);



/*
        double aU2 = square_euclid(aU);
        double aV2 = square_euclid(aV);
        double aUV = scal(aU,aV);
*/

         // De memoire, la + grande des VP de l'affinite
        // aCoefEtirReel = sqrt((aU2+aV2+sqrt(ElSquare(aU2-aV2)+4*ElSquare(aUV)))/2);

         // Version surfacique 

         aCoefEtirReel = ElAbs(aU.x * aV.y - aU.y * aV.x);


if (0)
{
static double aMinCER = 1e20;
static double aMaxCER = -1e20;
bool Modif = false;
if (aCoefEtirReel<aMinCER) 
{
     aMinCER = aCoefEtirReel;
     Modif = true;
}
if (aCoefEtirReel> aMaxCER) 
{
     aMaxCER = aCoefEtirReel;
     Modif = true;
}
if (Modif)
     std::cout << "COEFFF =[" << aMinCER << " " << aMaxCER  << "]\n";
}



/*
 std::cout << "aCoefEtirReel " << aCoefEtirReel << "\n";
if (aCoefEtirReel>1e10)
{
   std::cout << anIndexDef << m00 << " " << m10 << " " << m01 << " " << m11 << "\n";
   std::cout << u << " " << v  << "\n";
   std::cout << aU << " " << aV  << "\n";
getchar();
}
*/
        // aCoefEtire = ElMin(254,round_ni(aCoefEtirReel*mDynEtire));
        aCoefEtire = ElMax(1,ElMin(253,round_ni(mDynEtire/aCoefEtirReel)));
        if (aDet<0)
            aCoefEtire = 0;
     }  // WU : ENDSKIP

// if (Bug) std::cout << "aCoefEtireaCoefEtire " << aCoefEtirReel << " " <<  aCoefEtire << "\n";
                // BasculerUnTriangle(P00,P10,P11);
                // BasculerUnTriangle(P00,P11,P01);
     // if (aDet<0) return;

     if (aDet==0) return;  // WU=> is triangle bad oriented, not visible

     REAL zA = A3.z;
     REAL zB = B3.z;
     REAL zC = C3.z;

     //WU : P0,P1 => coordinate of integer rectangle containing A3,B3 and C3 (bounding box)
     Pt2di aP0 = round_down(Inf(A2,Inf(B2,C2)));
     aP0 = Sup(aP0,Pt2di(0,0));
     Pt2di aP1 = round_up(Sup(A2,Sup(B2,C2)));
     aP1 = Inf(aP1,mSzRes-Pt2di(1,1));

     // WU : This is used when we want to project images of attribute (like correl coeff) in more of the Z buff
     std::vector<double>  mAttrA;
     std::vector<double>  mAttrB;
     std::vector<double>  mAttrC;
     for (int aKA=0 ; aKA<(int)mImAttrIn.size() ; aKA++)
     {
         mAttrA.push_back(mImAttrIn[aKA]->GetR(A));
         mAttrB.push_back(mImAttrIn[aKA]->GetR(B));
         mAttrC.push_back(mImAttrIn[aKA]->GetR(C));
     }

     // WU : parse the bounding bow
     for (INT x=aP0.x ; x<= aP1.x ; x++)
     {
         for (INT y=aP0.y ; y<= aP1.y ; y++)
	 {
		 Pt2dr AP = Pt2dr(x,y)-A2;

         //WU :  compute barrrycentrique coordinate of a pixel in triangle A2,B2,C2
   
		 REAL aPdsB = (AP^AC) / aDet;
		 REAL aPdsC = (AB^AP) / aDet;
		 REAL aPdsA = 1 - aPdsB - aPdsC;

                  // WU : point belongs triangle iff the 3 coordinate are >=0 (Eps=>tiny number)
		 if ((aPdsA>-Eps) && (aPdsB>-Eps) && (aPdsC>-Eps))
		 {
                    // WU : interpolated value of Z
                    REAL4 aZ = (float) (zA *aPdsA  + zB* aPdsB + zC *aPdsC);

                    //  WU : if value is over previous one, it becomes the visible part 
                    if (aZ>mDataRes[y][x])
                    {
                         mDataRes[y][x] = aZ;
                         //  WU : other things are for additionnal attribute ....
                         mImTriInv.set(x,y,aDet<0);
                         if (aCoefEtire>=0)
                         {
                              mImEtirement.SetI(Pt2di(x,y),aCoefEtire);
                         }
                         double aMul =1.0;
                         if (mDynEtire>0)
                         {
                              aMul = ElMin(1.0,1/aCoefEtirReel);
                         }
                         for (int aKA=0 ; aKA<(int)mImAttrIn.size() ; aKA++)
                         {
                              
                              mImAttrOut[aKA]->SetR
                              (
                                 Pt2di(x,y),
                                 aMul * (aPdsA*mAttrA[aKA] + aPdsB*mAttrB[aKA] + aPdsC*mAttrC[aKA])
                              );
                         }
                    }
		 }
	 }
    }
}

Pt3dr cZBuffer::ToCoordInAbs(const Pt3dr & aPInDisc) const
{
   return  Pt3dr 
           (
               mOrigineIn.x+ aPInDisc.x*mStepIn.x,
               mOrigineIn.y+ aPInDisc.y*mStepIn.y,
               aPInDisc.z
           );
}

Pt3dr cZBuffer::ToCoordOutLoc(const Pt3dr & aPOutTer) const
{
   return Pt3dr
          (
	      (aPOutTer.x-mOrigineOut.x)/mStepOut.x -mOffet_Out_00.x,
	      (aPOutTer.y-mOrigineOut.y)/mStepOut.y -mOffet_Out_00.y,
	      aPOutTer.z
          );
}

Pt3dr cZBuffer::ProjDisc(const Pt3dr & aPInDisc) const
{
/*
   Pt3dr aPInTer 
         (
             mOrigineIn.x+ aPInDisc.x*mStepIn.x,
             mOrigineIn.y+ aPInDisc.y*mStepIn.y,
	     aPInDisc.z
	 );
*/
   Pt3dr aPInTer =  ToCoordInAbs(aPInDisc);
   Pt3dr aPOutTer = ProjTerrain(aPInTer);
   return  ToCoordOutLoc(aPOutTer);

/*
   return Pt3dr
          (
	      (aPOutTer.x-mOrigineOut.x)/mStepOut.x -mOffet_Out_00.x,
	      (aPOutTer.y-mOrigineOut.y)/mStepOut.y -mOffet_Out_00.y,
	      aPOutTer.z
	  );
*/
}

Pt3dr cZBuffer::ProjDisc(const Pt2di & aPInDisc,double * aPtrValZofXY) const
{
    if (mBufDone)
    {
       Pt2di aPDec = aPInDisc - mP0In;
       if (
                (aPDec.x<0)
             || (aPDec.y<0)
             || (aPDec.x>=mSzIn.x)
             || (aPDec.y>=mSzIn.y)

          )
       {
          std::cout << mP0In << aPInDisc  << mSzIn << "\n";
          ELISE_ASSERT(false,"Out in cZBuffer::ProjDisc");
       }
       Pt3dr   aP
               (
                   mDX3[aPDec.y][aPDec.x],
                   mDY3[aPDec.y][aPDec.x],
                   mDZ3[aPDec.y][aPDec.x]
               );
        return aP;
    }
 
    double  aValZofXY = ZofXY(aPInDisc);
    if (mIsRPC)
    {
         aValZofXY = ElMax(mZMinRPC,ElMin(mZMaxRPC,aValZofXY));
    }
    if (aPtrValZofXY) *aPtrValZofXY = aValZofXY;
    return ProjDisc(Pt3dr(aPInDisc.x,aPInDisc.y,aValZofXY));
    // return ProjDisc(Pt3dr(aPInDisc.x,aPInDisc.y,ZofXY(aPInDisc)));
}

Pt3dr cZBuffer::ProjReelle(const Pt2dr  & aPIn,bool & Ok) const
{
    if (mBufDone)
    {
       Pt2dr aPDec = aPIn - Pt2dr(mP0In);
       if ( ! mTX3.Rinside_bilin(aPDec))
       {
           Ok = false;
           return Pt3dr(0,0,0);
       }
       Ok = true;
       return Pt3dr ( mTX3.getr(aPDec), mTY3.getr(aPDec), mTZ3.getr(aPDec));
    }
    double aZ = ZInterpofXY(aPIn,Ok);

    if (!Ok)
       return Pt3dr(0,0,0);

    return ProjDisc(Pt3dr(aPIn.x,aPIn.y,aZ));
}





Pt3dr cZBuffer::InverseProjDisc(const Pt3dr & aPOutDisc) const
{
   Pt3dr aPOutTer 
         (
	     (aPOutDisc.x+mOffet_Out_00.x)*mStepOut.x +mOrigineOut.x,
	     (aPOutDisc.y+mOffet_Out_00.y)*mStepOut.y +mOrigineOut.y,
	     aPOutDisc.z
	 );

   Pt3dr aPInTer = InvProjTerrain(aPOutTer);

   return Pt3dr
          (
	      (aPInTer.x-mOrigineIn.x)/mStepIn.x,
	      (aPInTer.y-mOrigineIn.y)/mStepIn.y,
	      aPInTer.z
	  );
}

Pt3dr cZBuffer::InverseProjDisc(const Pt2di  & aP) const
{
     return InverseProjDisc(Pt3dr(aP.x,aP.y,mDataRes[aP.y][aP.x]));
}

Im2D_REAL4 cZBuffer::BasculerAndInterpoleInverse
           (
               Pt2di & aOffset_Out_00,
               Pt2di aP0In,
               Pt2di aP1In,
               float aDef,
	       bool * OkRes
           )

{
    bool OkBasc;
    Im2D_REAL4 aRes = Basculer(aOffset_Out_00,aP0In,aP1In,aDef,&OkBasc);

    if (OkRes) 
       *OkRes = OkBasc;

    if (! OkBasc)
    {
        ELISE_ASSERT(OkRes,"Cannot do cZBuffer::BasculerAndInterpoleInverse");
	return aRes;
    }

int aNBC = 0;
    Pt2di aP;
    for (aP.x=0 ; aP.x<mSzRes.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSzRes.y ; aP.y++)
	{
	     float & aZ1Out = mDataRes[aP.y][aP.x];
	     if (aZ1Out != aDef)
	     {
	       for (int aK=0 ; aK< 2 ; aK++)
	       {
	          bool OkInterp;
	          Pt3dr aP1 = InverseProjDisc(aP);
		  double aDZ1 = aP1.z-ZInterpofXY(Pt2dr(aP1.x,aP1.y),OkInterp);
                  if (OkInterp)
		  {
                      double aZ2Out = aZ1Out-aDZ1;
		      Pt3dr aP2 = InverseProjDisc(Pt3dr(aP.x,aP.y,aZ2Out));
		      double aDZ2 = aP2.z -ZInterpofXY(Pt2dr(aP2.x,aP2.y),OkInterp);

		      if (OkInterp && (ElAbs(aDZ1-aDZ2) >  mEpsIntInv))
		      {
		           double aZ3Out = aZ1Out - (aDZ1*(aZ2Out-aZ1Out))/(aDZ2-aDZ1);
		           Pt3dr aP3 = InverseProjDisc(Pt3dr(aP.x,aP.y,aZ3Out));
		           double aDZ3 = aP3.z -ZInterpofXY(Pt2dr(aP3.x,aP3.y),OkInterp);

		           if (OkInterp && (ElAbs(aDZ3) < ElAbs(aDZ1)))
		           {
aNBC++;
		               aZ1Out = (float)aZ3Out;
			       aDZ1 = aDZ3;
		           }

		           if (ElAbs(aDZ2) < ElAbs(aDZ1))
		           {
aNBC++;
		               aZ1Out = (float)aZ2Out;
			       aDZ1 = aDZ2;
		           }
		      }
		  }
	       }
	     }
	}
    }
    
std::cout << "NBC " << aNBC << " VS " << mSzRes.x * mSzRes.y << "\n";
    return aRes;
}


/*--------------------------------- */
/*     ANCIEN Z BUFFER              */
/*--------------------------------- */

/*
cZBuffer::cZBuffer(tMNT aMnt):
    pProjCur  (0),
    mMNT      (aMnt),
    mDMnt     (mMNT.data()),
    mSzMnt    (aMnt.sz())
{
}



Pt2dr cZBuffer::Mnt2Im(Pt2di aP)
{
   return pProjCur->Mnt2Im(aP,mDMnt[aP.y][aP.x]);
}

bool cZBuffer::PtMntDsIm(Pt2di aPMnt)
{
   Pt2dr aPIm = Mnt2Im(aPMnt);
   return 
               (aPIm.x >= 0)
           &&  (aPIm.y >= 0)
           &&  (aPIm.x < mSzIm.x)
           &&  (aPIm.y < mSzIm.y);
}

void cZBuffer::SetProj(cZBProj & aProj,Pt2di  aSz)
{
     mSzIm = aSz;
     pProjCur = &  aProj;

     INT aSzD = 50;
     mVBox.clear();

     for (INT anXG = 0 ; anXG<mSzMnt.x ; anXG += aSzD)
     {
         INT anX0 = anXG;
         INT anX1 = ElMin(mSzMnt.x-1,anX0+aSzD);
         for (INT anYG = 0 ; anYG<mSzMnt.y ; anYG += aSzD)
         {
             INT anY0 = anYG;
             INT anY1 = ElMin(mSzMnt.y-1,anY0+aSzD);


              bool Ok =      PtMntDsIm(Pt2di(anX0,anY0))
                         ||  PtMntDsIm(Pt2di(anX0,anY1-1))
                         ||  PtMntDsIm(Pt2di(anX1-1,anY0))
                         ||  PtMntDsIm(Pt2di(anX1-1,anY1-1));

             for (INT x = anX0 ; (x <anX1) && (!Ok) ; x++)
                 Ok =    Ok 
                      || PtMntDsIm(Pt2di(x,anY0))
                      || PtMntDsIm(Pt2di(x,anY1-1));

             for (INT y = anY0 ; (y <anY1) && (!Ok) ; y++)
                 Ok =    Ok 
                      || PtMntDsIm(Pt2di(anX0  ,y))
                      || PtMntDsIm(Pt2di(anX1-1,y));

             if (Ok)
                mVBox.push_back
                (
                    Box2di(Pt2di(anX0,anY0),Pt2di(anX1,anY1))
                );
	 }
     }
}


REAL Eps = 1e-7;

void  cZBuffer::ProjTri(Pt2di A,Pt2di B,Pt2di C)
{
     Pt2dr imA  =  Mnt2Im(A);
     Pt2dr imB  =  Mnt2Im(B);
     Pt2dr imC  =  Mnt2Im(C);

     Pt2dr imAB = imB-imA;
     Pt2dr imAC = imC-imA;
     REAL aDet = imAB^imAC;

     if (aDet<0)
        return;

     REAL zA = mDMnt[A.y][A.x];
     REAL zB = mDMnt[B.y][B.x];
     REAL zC = mDMnt[C.y][C.x];

     Pt2di aP0 = round_down(Inf(imA,Inf(imB,imC)));
     aP0 = Sup(aP0,Pt2di(0,0));
     Pt2di aP1 = round_up(Sup(imA,Sup(imB,imC)));
     aP1 = Inf(aP1,mSzIm-Pt2di(1,1));

     for (INT x=aP0.x ; x<= aP1.x ; x++)
         for (INT y=aP0.y ; y<= aP1.y ; y++)
	 {
		 Pt2dr AP = Pt2dr(x,y)-imA;

		 REAL aU = (AP^imAC) / aDet;
		 REAL aV = (imAB^AP) / aDet;
		 if ((aU>-Eps) && (aV>-Eps) && (aU+aV<1+Eps))
		 {
              REAL4 aZ = (float) (zA + (zB-zA) * aU + (zC-zA) *aV);
		      ElSetMax(mDMntPrIm[y][x],aZ);
		 }

	 }
}


cZBuffer::tMNT cZBuffer::ProjMnt()
{

     tMNT aRes(mSzIm.x,mSzIm.y);
     ELISE_COPY(aRes.all_pts(),-1e20,aRes.out());
     mDMntPrIm = aRes.data();

     for (INT aK = 0 ; aK<INT(mVBox.size()) ; aK++)
     {
         INT anX0 = mVBox[aK]._p0.x;
         INT anY0 = mVBox[aK]._p0.y;
         INT anX1 = mVBox[aK]._p1.x;
         INT anY1 = mVBox[aK]._p1.y;

         for (INT x= anX0 ; x<anX1 ; x++)
         {
            for (INT y= anY0 ; y<anY1 ; y++)
            {
                Pt2di P00(x,y);
                Pt2di P10(x+1,y);
                Pt2di P01(x,y+1);
                Pt2di P11(x+1,y+1);

                ProjTri(P00,P10,P11);
                ProjTri(P00,P11,P01);
            }
         }
    }
    return aRes;
}

void  cZBuffer::PartieCachees_Gen
      (
              tMNT aMntProj, REAL aStepXY, Pt2di aSzRes,
              tMNT * aResReal,
              Im2D_U_INT1  * aResQuant,REAL aStep,
              Im2D_Bits<1> * aResBin,REAL   aSeuil
      )
{
     ELISE_ASSERT(aMntProj.sz() == mSzIm,"cZBuffer::PartieCachees");
     REAL4 **  aDR = aResReal  ? aResReal->data()  : 0 ;
     U_INT1 ** aDU = aResQuant ? aResQuant->data() : 0 ;

     Im2D_Bits<1> aImNul(1,1);
     TIm2DBits<1> aTImBin(aResBin ? *aResBin :  aImNul);

     TIm2D<REAL4,REAL> aTMntInit(mMNT);
     TIm2D <REAL4,REAL> aTMntP(aMntProj);

     for (INT aK = 0 ; aK<INT(mVBox.size()) ; aK++)
     {
         INT anX0 = ElMax(0,round_ni(mVBox[aK]._p0.x/aStepXY));
         INT anY0 = ElMax(0,round_ni(mVBox[aK]._p0.y/aStepXY));
         INT anX1 = ElMin(aSzRes.x,round_ni(mVBox[aK]._p1.x/aStepXY));
         INT anY1 = ElMin(aSzRes.y,round_ni(mVBox[aK]._p1.y/aStepXY));
         for (INT x= anX0 ; x<anX1 ; x++)
         {
            for (INT y= anY0 ; y<anY1 ; y++)
            {
                Pt2di aP(x,y);
                Pt2dr aPr(x*aStepXY,y*aStepXY);
		if (aTMntInit.inside_bilin(aPr))
                {
                   REAL aZ = aTMntInit.getr(aPr);
                   Pt2dr aQr = pProjCur->Mnt2Im(aPr,aZ);

                   if (aTMntP.inside_bilin(aQr))
                   {
                      REAL aVal = ElMax(0.0,aTMntP.getr(aQr)-aZ);
                      if (aDR)
                         aDR[y][x] = (float) aVal;
                      if (aDU)
                         aDU[y][x] = ElMin(255,round_ni(aVal/aStep));
                      if (aResBin)
                         aTImBin.oset(aP,aVal<aSeuil);
		   }
                }
	    }
	 }
     }
}

cZBuffer::tMNT   cZBuffer::PartieCachees (tMNT aMntProj,REAL aStepXY)
{
     tMNT aRes(round_ni(mSzMnt.x/aStepXY),round_ni(mSzMnt.y/aStepXY),0.0);

     PartieCachees_Gen ( aMntProj,aStepXY,aRes.sz(), &aRes, 0,0, 0,0);
     return aRes;
}

Im2D_U_INT1   cZBuffer::PartieCachees_Quant (tMNT aMntProj,REAL aStep,REAL aStepXY)
{
     Im2D_U_INT1 aRes(round_ni(mSzMnt.x/aStepXY),round_ni(mSzMnt.y/aStepXY),0);

     PartieCachees_Gen(aMntProj,aStepXY,aRes.sz(),0,&aRes,aStep,0,0);
     return aRes;
}


Im2D_Bits<1>   cZBuffer::PartieCachees_Bin (tMNT aMntProj,REAL aSeuil,REAL aStepXY)
{
     Im2D_Bits<1> aRes(round_ni(mSzMnt.x/aStepXY),round_ni(mSzMnt.y/aStepXY),0);

     PartieCachees_Gen(aMntProj,aStepXY,aRes.sz(),0,0,0,&aRes,aSeuil);
     return aRes;
}
*/


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

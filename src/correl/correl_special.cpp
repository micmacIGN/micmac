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



static void VerifIm(Pt2di aSz,   Pt2di aP0, Pt2di aP1, INT aNb)
{
       ELISE_ASSERT
       (
              (aP0.x>=aNb) && (aP0.y>=aNb)
           && (aP1.x<aSz.x- aNb) && (aP1.y<aSz.y- aNb),
           "Bas Size in SomSquare"
       );
}



void Somme_Line
     (
          Im2D_INT4 anISom,
          Im2D_INT4 aIBuf,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     )
{
    INT aNbCol =  aP1.x-aP0.x;
    INT4 ** aSom = anISom.data();
    INT4 ** aBuf = aIBuf.data();


    MEM_RAZ(aBuf[aP0.y]+aP0.x,aNbCol);

    for (INT anY = aP0.y-aNb; anY<= aP0.y+aNb ; anY++)
    {
	INT * aLBuf = aBuf[aP0.y] +  aP0.x;
	INT * aLSom = aSom[anY]   +  aP0.x;

        for (INT aNbX = aNbCol; aNbX ; aNbX--)
           *(aLBuf++) += *(aLSom++);
    }


    for (INT anY = aP0.y+1 ; anY < aP1.y ; anY++)
    {
	INT * aLPrec = aBuf[anY-1]     +  aP0.x;
	INT * aLBuf  = aBuf[anY]       +  aP0.x;
	INT * aBack  = aSom[anY-aNb-1] +  aP0.x;
	INT * aFront = aSom[anY+aNb]   +  aP0.x;

        for (INT aNbX = aNbCol ; aNbX ; aNbX--)
	    *(aLBuf++) = *(aLPrec++)+ *(aFront++) - *(aBack++);
    }

    for (INT anY = aP0.y ; anY < aP1.y ; anY++)
        memcpy
        (
           aSom[anY]+aP0.x,
           aBuf[anY]+aP0.x,
           aNbCol*sizeof(**aSom)
        );
}


void Somme_12_2_22
     (
          Im2D_INT4 aSom12,
          Im2D_INT4 aSom2,
          Im2D_INT4 aSom22,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Im2D_U_INT1 anIm2,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     )
{

       VerifIm(aSom12.sz(),aP0,aP1,aNb);
       VerifIm(aSom2.sz() ,aP0,aP1,aNb);
       VerifIm(aSom22.sz(),aP0,aP1,aNb);
       VerifIm(aBuf.sz()  ,aP0,aP1,aNb);
       VerifIm(anIm1.sz()  ,aP0,aP1,aNb);
       VerifIm(anIm2.sz()  ,aP0,aP1,aNb);

       INT4 ** aS12 = aSom12.data();
       INT4 ** aS2  = aSom2.data();
       INT4 ** aS22 = aSom22.data();
       U_INT1 ** aI1  = anIm1.data();
       U_INT1 ** aI2  = anIm2.data();

       for (INT anY = aP0.y-aNb; anY<aP1.y+aNb ; anY++)
       {
	    INT * aLS12 = aS12[anY] + aP0.x;
	    INT * aLS2  = aS2[anY]  + aP0.x;
	    INT * aLS22 = aS22[anY] + aP0.x;


	    U_INT1 * aBack1 = aI1[anY]+ aP0.x -aNb;
	    U_INT1 * aBack2 = aI2[anY]+ aP0.x -aNb;

	    U_INT1 * aFront1 =  aBack1;
	    U_INT1 * aFront2 =  aBack2;

	    * aLS12 = 0;
	    * aLS2  = 0;
	    * aLS22 = 0;

            for (INT ad=-aNb; ad<= aNb ; ad++)
	    {
	       * aLS12 += *aFront1 *  *aFront2;
	       * aLS2  += *aFront2;
	       * aLS22 += *aFront2 *  *aFront2;
		aFront1++;
		aFront2++;
	    }
	    INT aF2,aB2;
	    for (INT aNbX = aP1.x-aP0.x-1; aNbX ; aNbX--)
	    {
		aF2 =  *(aFront2++);
		aB2 =  *(aBack2++);
		aLS12[1] = *aLS12 + *(aFront1++)* aF2 -*(aBack1++)* aB2;
		aLS2[1]  = *aLS2  + aF2-aB2;
		aLS22[1] = *aLS22 + aF2*aF2-aB2*aB2;
		aLS12++; aLS2++; aLS22++;
	    }
     }

     Somme_Line(aSom12,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom2,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom22,aBuf,aP0,aP1,aNb);
}

void Somme__1_11
     (
          Im2D_INT4 aSom,
          Im2D_INT4 aSom1,
          Im2D_INT4 aSom11,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     )
{

       VerifIm(aSom.sz(),aP0,aP1,aNb);
       VerifIm(aSom1.sz() ,aP0,aP1,aNb);
       VerifIm(aSom11.sz(),aP0,aP1,aNb);
       VerifIm(anIm1.sz()  ,aP0,aP1,aNb);
       VerifIm(aBuf.sz()  ,aP0,aP1,aNb);

       {
	  INT aNbVois = ElSquare(1+2*aNb);
          for (INT anY = aP0.y ; anY < aP1.y ; anY++)
	  {
               INT * aLS = aSom.data()[anY]+aP0.x;
	       for (INT aNbX = aP1.x-aP0.x; aNbX ; aNbX--)
		    *(aLS++) = aNbVois;
	  }
       }


       INT4 ** aS1  = aSom1.data();
       INT4 ** aS11 = aSom11.data();
       U_INT1 ** aI1  = anIm1.data();

       for (INT anY = aP0.y-aNb; anY<aP1.y+aNb ; anY++)
       {
	    INT * aLS1  = aS1[anY]  + aP0.x;
	    INT * aLS11 = aS11[anY] + aP0.x;


	    U_INT1 * aBack1 = aI1[anY]+ aP0.x -aNb;
	    U_INT1 * aFront1 =  aBack1;

	    * aLS1  = 0;
	    * aLS11 = 0;

            for (INT ad=-aNb; ad<= aNb ; ad++)
	    {
	       * aLS1  += *aFront1;
	       * aLS11 += *aFront1 *  *aFront1;
		aFront1++;
	    }
	    INT aF1,aB1;
	    for (INT aNbX = aP1.x-aP0.x-1; aNbX ; aNbX--)
	    {
		aF1 =  *(aFront1++);
		aB1 =  *(aBack1++);
		aLS1[1]  = *aLS1  + aF1-aB1;
		aLS11[1] = *aLS11 + aF1*aF1-aB1*aB1;
		aLS1++; aLS11++;
	    }
     }

     Somme_Line(aSom1,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom11,aBuf,aP0,aP1,aNb);
}





void Somme__1_2_11_12_22
     (
          Im2D_INT4 aSom,
          Im2D_INT4 aSom1,
          Im2D_INT4 aSom2,
          Im2D_INT4 aSom11,
          Im2D_INT4 aSom12,
          Im2D_INT4 aSom22,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Im2D_U_INT1 anIm2,
          Im2D_U_INT1 aPond,
          Pt2di       aP0,
          Pt2di       aP1,
          INT         aNb
     )
{

       VerifIm(aSom.sz(),aP0,aP1,aNb);
       VerifIm(aSom1.sz(),aP0,aP1,aNb);
       VerifIm(aSom2.sz(),aP0,aP1,aNb);
       VerifIm(aSom11.sz(),aP0,aP1,aNb);
       VerifIm(aSom12.sz(),aP0,aP1,aNb);
       VerifIm(aSom22.sz(),aP0,aP1,aNb);
       VerifIm(aBuf.sz()  ,aP0,aP1,aNb);
       VerifIm(anIm1.sz() ,aP0,aP1,aNb);
       VerifIm(anIm2.sz() ,aP0,aP1,aNb);
       VerifIm(aPond.sz() ,aP0,aP1,aNb);

       INT4 ** aS   = aSom.data();
       INT4 ** aS1  = aSom1.data();
       INT4 ** aS2  = aSom2.data();
       INT4 ** aS11 = aSom11.data();
       INT4 ** aS12 = aSom12.data();
       INT4 ** aS22 = aSom22.data();

       U_INT1 ** aI1  = anIm1.data();
       U_INT1 ** aI2  = anIm2.data();
       U_INT1 ** aP   = aPond.data();

       for (INT anY = aP0.y-aNb; anY<aP1.y+aNb ; anY++)
       {

	    INT * aLS   =  aS[anY]  + aP0.x;
	    INT * aLS1  = aS1[anY]  + aP0.x;
	    INT * aLS2  = aS2[anY]  + aP0.x;
	    INT * aLS11 = aS11[anY] + aP0.x;
	    INT * aLS12 = aS12[anY] + aP0.x;
	    INT * aLS22 = aS22[anY] + aP0.x;


	    U_INT1 * aBack1 = aI1[anY] + aP0.x -aNb;
	    U_INT1 * aBack2 = aI2[anY] + aP0.x -aNb;
	    U_INT1 * aBackP =  aP[anY] + aP0.x -aNb;

	    U_INT1 * aFront1 =  aBack1;
	    U_INT1 * aFront2 =  aBack2;
	    U_INT1 * aFrontP =  aBackP;

	    * aLS   = 0;
	    * aLS1  = 0;
	    * aLS2  = 0;
	    * aLS11 = 0;
	    * aLS12 = 0;
	    * aLS22 = 0;

            for (INT ad=-aNb; ad<= aNb ; ad++)
	    {
               if (*aFrontP)
               {
                   (*aLS)++;
	           *aLS1  += *aFront1;
	           *aLS2  += *aFront2;
	           *aLS11 += *aFront1 *  *aFront1;
	           *aLS12 += *aFront1 *  *aFront2;
	           *aLS22 += *aFront2 *  *aFront2;
                }
		aFront1++;
		aFront2++;
		aFrontP++;
	    }
	    INT aV1,aV2;
	    for (INT aNbX = aP1.x-aP0.x-1; aNbX ; aNbX--)
	    {
                if (*(aBackP++))
                {
                     aV1 = *(aBack1++);
                     aV2 = *(aBack2++);

                     aLS[1]  = *aLS    - 1;
                     aLS1[1] = *aLS1   - aV1;
                     aLS2[1] = *aLS2   - aV2;
                     aLS11[1] = *aLS11 - aV1*aV1;
                     aLS12[1] = *aLS12 - aV1*aV2;
                     aLS22[1] = *aLS22 - aV2*aV2;
                }
                else
                {
                     aBack1++;
                     aBack2++;
                     aLS[1] = *aLS;
                     aLS1[1] = *aLS1;
                     aLS2[1] = *aLS2;
                     aLS11[1] = *aLS11;
                     aLS12[1] = *aLS12;
                     aLS22[1] = *aLS22;
                }

                if (*(aFrontP++))
                {
                     aV1 = *(aFront1++);
                     aV2 = *(aFront2++);

                     aLS[1]  +=  1;
                     aLS1[1] +=  aV1;
                     aLS2[1] +=  aV2;
                     aLS11[1] += aV1*aV1;
                     aLS12[1] += aV1*aV2;
                     aLS22[1] += aV2*aV2;
                }
                else
                {
                     aFront1++;
                     aFront2++;
                }

		aLS++    ;  aLS1++   ;  aLS2++;
		aLS11++  ;  aLS12++  ;  aLS22++;
	    }

     }

     Somme_Line(aSom,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom1,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom2,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom11,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom12,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom22,aBuf,aP0,aP1,aNb);
}







void EliseCorrel2D::ComputeICorrel
     (
          Im2D_INT4  aSom12,
          Im2D_INT4  aBuf,
          Im2D_REAL4 aRes,
          Pt2di      aP0I1,
          Pt2di      aP0I2,
          Pt2di      aSz
     )
{

       VerifIm(mI1.sz(),aP0I1,aP0I1+aSz,mSzV);
       VerifIm(mI2.sz(),aP0I2,aP0I2+aSz,mSzV);

       INT4 ** aS12 = aSom12.data();

       for (INT aDY = -mSzV; aDY<aSz.y+mSzV ; aDY++)
       {
            INT anY1 = aDY+aP0I1.y;
            INT anY2 = aDY+aP0I2.y;

	    INT * aLS12 = aS12[anY2] + aP0I2.x;


	    U_INT1 * aBack1 = mDataI1[anY1]+ aP0I1.x -mSzV;
	    U_INT1 * aBack2 = mDataI2[anY2]+ aP0I2.x -mSzV;

	    U_INT1 * aFront1 =  aBack1;
	    U_INT1 * aFront2 =  aBack2;

	    * aLS12 = 0;

            for (INT ad=-mSzV; ad<= mSzV ; ad++)
	    {
	       * aLS12 += *aFront1 *  *aFront2;
		aFront1++;
		aFront2++;
	    }
	    for (INT aNbX = aSz.x-1; aNbX ; aNbX--)
	    {
		aLS12[1] = *aLS12 + *(aFront1++)* *(aFront2++)
                                 -  *(aBack1++)* *(aBack2++);
		aLS12++;
	    }
     }

     Somme_Line(aSom12,aBuf,aP0I2,aP0I2+aSz,mSzV);

     INT aY1I2 = aP0I2.y+aSz.y;
     INT aX1I2 = aP0I2.x+aSz.x;
     REAL4 ** aDataRes = aRes.data();
     INT aNbV = ElSquare(1+2*mSzV);
     Pt2di aRDec = aP0I1 - aP0I2;
     for (INT aY2=aP0I2.y, aY1 = aP0I1.y ; aY2 <aY1I2  ; aY2++,aY1++)
     {
         for (INT aX2=aP0I2.x, aX1=aP0I1.x ; aX2<aX1I2  ; aX2++,aX1++)
         {
              REAL C2 = (aS12[aY2][aX2]/REAL(aNbV) -mDataS1[aY1][aX1]*mDataS2[aY2][aX2]);
              C2 /= sqrt(mDataS11[aY1][aX1]*mDataS22[aY2][aX2]);

              if (C2 > aDataRes[aY2][aX2] )
              {
                 aDataRes[aY2][aX2] = (float) C2;
                 mDec.SetDec(Pt2di(aX2,aY2),Pt2dr(aRDec));
                 // aDecX[aY2][aX2] =
                 // aDecY[aY2][aX2] =
                 if (mD1GRI2)
                 {
                        mD1GRI2[aY2][aX2] =    mDataS2[aY2][aX2]
                                             +   (mDataI1[aY1][aX1]-mDataS1[aY1][aX1])
                                               * sqrt(mDataS22[aY2][aX2]/mDataS11[aY1][aX1]);
                 }
              }

         }
     }
}


void EliseCorrel2D::ComputeCorrelMax(Pt2di aDec0,INT anIncert)
{
    ELISE_COPY(mCorrelMax.all_pts(),-1e7,mCorrelMax.out());

    Im2D_INT4  aSom12(mSzIm.x,mSzIm.y);
    Im2D_INT4  aBuf  (mSzIm.x,mSzIm.y);
    Im2D_REAL4 aRes  (mSzIm.x,mSzIm.y);

    for (INT x=-anIncert ; x<=anIncert ; x++)
    {
        // if (x%10==0)
        //    cout << "EliseCorrel2D::ComputeCorrelMax " << x << " " << anIncert << "\n";

        for (INT y=-anIncert ; y<=anIncert ; y++)
        {
              Pt2di aVois(mSzV,mSzV);

              Pt2di aDec = aDec0 + Pt2di(x,y);

              Pt2di aP0I2 =  Sup(aVois,-aDec);
              Pt2di aP0I1 =  Sup(aVois,aP0I2+aDec);

              aP0I2 =  Sup(aVois,aP0I1-aDec);
              aP0I1 =  Sup(aVois,aP0I2+aDec);



              Pt2di aSz = mSzIm-aVois-Pt2di(1,1)-Sup(aP0I2,aP0I1);
              if ((aSz.x >0) && (aSz.y>0))
              {
                  ComputeICorrel(aSom12,aBuf,aRes,aP0I1,aP0I2,aSz);
              }
        }
    }
}





void Somme_Sup0_1_11
     (
          Im2D_INT4 aSom0,
          Im2D_INT4 aSom1,
          Im2D_INT4 aSom11,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     )
{

       VerifIm(aSom0.sz(),aP0,aP1,aNb);
       VerifIm(aSom1.sz() ,aP0,aP1,aNb);
       VerifIm(aSom11.sz(),aP0,aP1,aNb);
       VerifIm(anIm1.sz()  ,aP0,aP1,aNb);
       VerifIm(aBuf.sz()  ,aP0,aP1,aNb);


       INT4 ** aS0  = aSom0.data();
       INT4 ** aS1  = aSom1.data();
       INT4 ** aS11 = aSom11.data();
       U_INT1 ** aI1  = anIm1.data();

       for (INT anY = aP0.y-aNb; anY<aP1.y+aNb ; anY++)
       {
	    INT * aLS0  = aS0[anY]  + aP0.x;
	    INT * aLS1  = aS1[anY]  + aP0.x;
	    INT * aLS11 = aS11[anY] + aP0.x;


	    U_INT1 * aBack1 = aI1[anY]+ aP0.x -aNb;
	    U_INT1 * aFront1 =  aBack1;

	    * aLS0  = 0;
	    * aLS1  = 0;
	    * aLS11 = 0;

            for (INT ad=-aNb; ad<= aNb ; ad++)
	    {
	       * aLS0  += (*aFront1>0);
	       * aLS1  += *aFront1;
	       * aLS11 += *aFront1 *  *aFront1;
		aFront1++;
	    }
	    INT aF1,aB1;
	    for (INT aNbX = aP1.x-aP0.x-1; aNbX ; aNbX--)
	    {
		aF1 =  *(aFront1++);
		aB1 =  *(aBack1++);
		aLS0[1]  = *aLS0  + (aF1>0)-(aB1>0);
		aLS1[1]  = *aLS1  + aF1-aB1;
		aLS11[1] = *aLS11 + aF1*aF1-aB1*aB1;
		aLS1++; aLS11++;aLS0++;
	    }
     }

     Somme_Line(aSom0,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom1,aBuf,aP0,aP1,aNb);
     Somme_Line(aSom11,aBuf,aP0,aP1,aNb);
}


void Somme_12
     (
          Im2D_INT4 aSom12,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Im2D_U_INT1 anIm2,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     )
{

       VerifIm(aSom12.sz(),aP0,aP1,aNb);
       VerifIm(aBuf.sz()  ,aP0,aP1,aNb);
       VerifIm(anIm1.sz()  ,aP0,aP1,aNb);
       VerifIm(anIm2.sz()  ,aP0,aP1,aNb);

       INT4 ** aS12 = aSom12.data();
       U_INT1 ** aI1  = anIm1.data();
       U_INT1 ** aI2  = anIm2.data();

       for (INT anY = aP0.y-aNb; anY<aP1.y+aNb ; anY++)
       {
	    INT * aLS12 = aS12[anY] + aP0.x;

	    U_INT1 * aBack1 = aI1[anY]+ aP0.x -aNb;
	    U_INT1 * aBack2 = aI2[anY]+ aP0.x -aNb;

	    U_INT1 * aFront1 =  aBack1;
	    U_INT1 * aFront2 =  aBack2;

	    * aLS12 = 0;

            for (INT ad=-aNb; ad<= aNb ; ad++)
	    {
	       * aLS12 += *(aFront1++) *  *(aFront2++);
	    }
	    for (INT aNbX = aP1.x-aP0.x-1; aNbX ; aNbX--)
	    {
		aLS12[1] =   *aLS12
                           + *(aFront1++)* *(aFront2++)
			   - *(aBack1++)* *(aBack2++);
		aLS12++;
	    }
     }

     Somme_Line(aSom12,aBuf,aP0,aP1,aNb);
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

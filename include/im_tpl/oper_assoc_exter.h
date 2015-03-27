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



#ifndef _OPER_ASSOC_EXTERN_H_
#define _OPER_ASSOC_EXTERN_H_

#include <complex>


/*


   cTplOpbBufImage<tArg> : classe Template d'Operateur de Bufferisation sur
   les images.

    Cette classe permet une implementation generique des algorithmes
    de type "somme rapide", "correlation rapide" , "correlation differentielle rapide"
    etc ....

   Voir aussi "ex_oper_assoc_exter.h" qui contient trois exemples complets
   d'utilisation :

   *  TestSomRapide : utilisation pour un filtre moyenneur rapide

   *  TestCorrel :  utilisation pour une correlatiob rapide

   *  TestSomIteree :  utilisation pour un filtre moyenneur en cascade (la moyenne de la moyenne) rapide.




                      !!!!!!!!!!!!!!!!!!!!!!

    Les types que la classe tArg doit definir  :
  
         un type tArg::tElem  qui definit les entrees de l'operateur

         un type tArg::tCumul qui definit le type necessaire
         pour representer l'aggregation des entrees

	 tArg lui meme a des pre-requis

    On designe par la suite tElem et tCumul les 2 types.


                      !!!!!!!!!!!!!!!!!!!!!!
tElem :
=====
    Il n'y a aucun pre-requis sur le type tArg::tElem;

 exemples:
    
    Si le filtre est un simple filtre de somme rapide,
    les entrees sont des valeurs scalaires et un type
    int ou double fera tres bien l'affaire
          
    Si le filtre est un filtre de correlation rapide,
    les entrees sont des couples de valeur scalaires
    (les valeurs de chacune des deux images a correler)
    et n'import quel type capable de stocker ces
    deux valeur fera l'affaire. Voir par exemple
    le type cVal2Image dans "include/im_tpl/ex_oper_assoc_exter.h"
    (mais, au prix de la lisibilite, on pourrait utiliser un type
     comme std::complex<int> ).

    

tCumul :
======
     
    En pre -requis le type tArg::tCumul doit definir 
    les fonctions suivantes :

             tCumul::tCumul();
             void AddElem(int aSigne,const tElem &);
             void AddCumul(int aSigne,const tCumul & );

     Au risque d'etre un peu pedant, le plus simple
     est d'utiliser le langage des groupes pour caracteriser
     les pre-requis qui font que la definition de ces fonctions
     donnera un comportement coherent.

     1) tCumul doit avoir une structure de groupe commutatif G.
     Notons a+b l'operation de groupe.
     Il doit exister une application canonique F de tElem vers tCumul
 
     Dans l'exemple de correlation rapide de 
     "include/im_tpl/ex_oper_assoc_exter.h", le type  tCumul
     est cCumulVarCov, il permet de memoriser les moments
     d'ordre 0,1 et 2. 

     La structure de groupe vient de l'operation qui consiste
     betement a additionner les moments.
      

  
     2) Le constructeur par defaut tCumul() doit creer
     l'element neutre de G

     3)  X.AddCumul(int aSigne,const tCumul & Y)

         Doit effectuer   X <- X+Y  (Signe = 1 )
	                  X <- X-Y  (Signe =-1 )

     4)    X. AddElem(int aSigne,const tElem & anEl)
	   
	Doit effectuer   X <- X+F(anEl)  (Signe = 1 )
	                 X <- X-F(anEl)  (Signe =-1 )



tArg -Prerequis:
===============
      Doit definir les fonction suivantes :

      void Init(const std::complex<int> & aP,tElem & anEl)
      indique comment creer la valeur en un point P a partir de l'infomation
      contenu dans tArg; typiquemnt si anEl est un type
      scalaire, Init remplira anEl avec la valeur de
      "l'image" en aP.


      void UseAggreg(const std::complex<int> & aP,const cCumulVarCov & aCVC)
      Dans l'hypothese d'une utilisation par "DoIt", c'est un Call-Back
      rappele pour chaque point avec la valeur cumulee en ce point
      (typiquemnt pour la correlation on recupera les moments en 
       un point, restera a l'utiliser pour calculer le coefficient
       et effectuer l'action qui va bien).

      void OnNewLine(int anY)  :  appele a chaque nouvelle
      ligne, pas fondamental, permet de suivre l'avancement





Constructeur  :
===============
             cTplOpbBufImage<tArg>
             (
                  tArg &  anArg,
                  const std::complex<int> & aP0Im,
                  const std::complex<int> & aP1Im,
                  const std::complex<int> & aP0Win,
                  const std::complex<int> & aP1Win
             )  :


   anArg contient l'information qui permettra
   de calculer Elem et de reagir a Cumul


   aP0Im aP1Im : definit le rectangle englobant sur
   lequels on souaite evaluer "l'operateur rapide"

   aP0Win aP1Win  : definit la fentre, elle n'est ni
   necessairement centre ni necessairement identique
   en X et en Y.  C'est cependant souvent le cas et
   on utilsera alors qqch comme :


         (...
             std::complex<int>(-aSzV,-aSzV),
             std::complex<int>(aSzV,aSzV)
	 );

Utilisation (1):
================
Une fois l'objet construit, on peut utiliser simplement

   DoIt();

L'effet sera de calculer tous les tCumul et de rappeler
le void UseAggreg(const std::complex<int> & aP,const cCumulVarCov & aCVC)
pour laisser l'utilisateur le manipuler.



Utilisation (2):
================

Une autre solution, plus facile a utiliser lorsque
l'operateur rapide est utilise pour etre reinjecte
comme entree d'un filtrage (par ex dans un utilisation
en cascade de filtrage rapide comme dans l'exemple TestSomIteree),
est de passer par :

             tCumul*   GetNext();

Il renvoie les elements successifs de cumul en balayant
le recangle selon l'orde dit "video", et renvoie (tCumul *)NULL
a la fin (balayage identique pour   DoIt()).

On peut dans ce cas acceder  au point courant calcule
par CurPtOut();

                      !!!!!!!!!!!!!!!!!!!!!!

Memoire temporaire utilisee en propre:

   tElem * (aP1Im.x -aP0Im.x + aP1Win.x-aP0Win.x) * (aP1Win.y-aP0Win.y)

   tCumul * (aP1Im.x -aP0Im.x)
   
                      !!!!!!!!!!!!!!!!!!!!!!

Autre caracteristique :

   Une caracteristique "amusante" de cette implementation
 est que (pour peu que la vignette contiennent le point (0,0))
 la meme image peut etre utilisée en entree et en sortie sans
 effet de bord (il se trouve que l'algorithme conserve pour
 ses besoins propres un bandeau autour des valeur courrament
 utilisées en entrée).

*/



template <class tArg> 
class  cTplOpbBufImage
{
   public :
     typedef typename tArg::tElem  tElem;
     typedef tElem *               tElPtr;
     typedef typename tArg::tCumul tCumul;
    // Conventions BoxIm aP0Im, aP1Im  : exclut P1
    //             BoxWin aP0Win,aP1Win : inclut P1
             cTplOpbBufImage 
             (
                  tArg &  anArg,
                  const std::complex<int> & aP0Im,
                  const std::complex<int> & aP1Im,
                  const std::complex<int> & aP0Win,
                  const std::complex<int> & aP1Win
             )  :
                mArg         (anArg),

                mY0Out       (aP0Im.imag()),
                mYCurOut     (mY0Out-1),
                mY1Out       (aP1Im.imag()),

                mY0CurIn     (aP0Win.imag()+aP0Im.imag()),
                mY1CurIn     (aP1Win.imag()+aP0Im.imag()),
                mNbYIn       (mY1CurIn-mY0CurIn+1),

                mX0CurOut    (aP0Im.real()),
                mX1CurOut    (aP1Im.real()),
                mCurXOut     (mX1CurOut-1),
                mNbXOut      (mX1CurOut-mX0CurOut),

                mX0CurIn     (aP0Win.real()+aP0Im.real()),
                mX1CurIn     (aP1Win.real()+aP1Im.real()),
                mNbXIn       (mX1CurIn-mX0CurIn),
                mNbWX        (aP1Win.real()-aP0Win.real()),

                mMatElem     ((new tElPtr [mNbYIn])-mY0CurIn),
                mVCumul      ((new tCumul [mNbXOut])-mX0CurOut)
             {
                 for (int aYIn=mY0CurIn ; aYIn<= mY1CurIn ; aYIn++)
                 {
                     mMatElem[aYIn] = (new tElem [mNbXIn]) -mX0CurIn;
                     if (aYIn != mY1CurIn)
                        AddNewLine(aYIn);
                 }
             }
             int  XOutDebLigne() const {return mX0CurOut;}
             int  XOutFinLigne() const {return mX1CurOut;}
             ~cTplOpbBufImage()
             {
                 for (int aYIn=mY0CurIn; aYIn<=mY1CurIn ; aYIn++)
                     delete [] (mMatElem[aYIn]+mX0CurIn);
                 delete [] (mMatElem+mY0CurIn);
                 delete [] (mVCumul+mX0CurOut);
             }
             void DoIt()
             {
                  mYCurOut++;
                  for (;mYCurOut<mY1Out ; mYCurOut++)
                  {
                      BeginOfLine();
                      for (mCurXOut=mX0CurOut ; mCurXOut<mX1CurOut ; mCurXOut++)
                      {
                          mArg.UseAggreg(CurPtOut(),mVCumul[mCurXOut]);
                      }
                      EndOfLine();
                  }
             }
             
             std::complex<int> CurPtOut()  const
             {
                 return std::complex<int>(mCurXOut,mYCurOut);
             }
             int CurXOut() const  {return mCurXOut;}
             int CurYOut() const  {return mYCurOut;}


             inline tCumul *  GetNext()
             {
                 if (mCurXOut ==  mX1CurOut-1)
                 {
                      if (mYCurOut >= mY0Out)
                      {
                          EndOfLine();
                      }
                      mYCurOut++;
                      mCurXOut = mX0CurOut;
                      if (mYCurOut==mY1Out)
                         return 0;
                     BeginOfLine();
                 }
                 else 
                    mCurXOut++;
                 return &mVCumul[mCurXOut];
             }

       private :

             void BeginOfLine ()
             {
                      AddNewLine(mY1CurIn);
                      mArg.OnNewLine(mYCurOut);
             }
             void EndOfLine ()
             {
                      CumulLine(mY0CurIn,-1);
                      // Permuation circulaire du "Buffer de lignes"
                      tElPtr aL0 = mMatElem[mY0CurIn];
                      for (int aYIn = mY0CurIn  ; aYIn<mY1CurIn ; aYIn++)
                          mMatElem[aYIn] = mMatElem[aYIn+1];
                       mMatElem[mY1CurIn] = aL0;

                      mY0CurIn++;
                      mY1CurIn++;
                      mMatElem--;
             }


             cTplOpbBufImage(const cTplOpbBufImage &); // Non Implemente
             void AddNewLine(int aYIn)
             {
                  tElem * aL = mMatElem[aYIn];
                  
                   for (int anX = mX0CurIn; anX<mX1CurIn  ; anX++)
                     mArg.Init(std::complex<int>(anX,aYIn),aL[anX]);
                  CumulLine(aYIn,1);
             }

             void CumulLine(int aYIn,int aSigne)
             {
                 tElem * aL = mMatElem[aYIn];
                 tCumul  anAccum;
                 for (int aXIn = mX0CurIn ; aXIn < mX0CurIn+mNbWX; aXIn++)
                     anAccum.AddElem(1,aL[aXIn]);

                 int aXArIn = mX0CurIn;
                 int aXAvIn = mX0CurIn+mNbWX;

                 for (int aXOut=mX0CurOut ; aXOut<mX1CurOut ; aXOut++)
                 {
                     anAccum.AddElem(1,aL[aXAvIn]);
                     mVCumul[aXOut].AddCumul(aSigne,anAccum);
                     anAccum.AddElem(-1,aL[aXArIn]);
                     aXArIn++;
                     aXAvIn++;
                 }
             }


            tArg &    mArg;

            int       mY0Out;
            int       mYCurOut;
            int       mY1Out;

            int       mY0CurIn;
            int       mY1CurIn;
            int       mNbYIn;

            int       mX0CurOut;
            int       mX1CurOut;
            int       mCurXOut;
            int       mNbXOut;

            int       mX0CurIn;
            int       mX1CurIn;
            int       mNbXIn;
            int       mNbWX;
            int       mCurYIn;

            tElem **  mMatElem;
            tCumul *  mVCumul;
};

#endif // _OPER_ASSOC_EXTERN_H_




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

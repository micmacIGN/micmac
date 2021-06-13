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
// #include "anag_all.h"

#include "CmpMNT.h"

namespace NS_CmpMNT
{

cAppliCmpMnt::cAppliCmpMnt(const cCompareMNT & aCmpMnt) :
   mArg    (aCmpMnt),
   mSz     (0,0),
   mImDiff (1,1),
   mCout   (mArg.NameFileRes().c_str(),ios::out)
{
   int aK=0;
   for 
   (
       std::list<cMNT2Cmp>::const_iterator itM = mArg.MNT2Cmp().begin();
       itM != mArg.MNT2Cmp().end();
       itM++
   )
   {
      cOneMnt * aPrec = mMnts.empty() ? 0 : mMnts[0];
      mMnts.push_back(new cOneMnt(*this,*itM,aK,mArg,aPrec));
      if (aK==0)
      {
         mSz = mMnts[0]->SzFile();
      }
      aK++;
   }

   aK=0;
   for 
   (
       std::list<cZoneCmpMnt>::const_iterator itM = mArg.ZoneCmpMnt().begin();
       itM != mArg.ZoneCmpMnt().end();
       itM++
   )
   {
      mZones.push_back(new cOneZoneMnt(*this,*itM,mArg));
   }
}

cAppliCmpMnt::~cAppliCmpMnt()
{
   mCout.close();
}

cAppliCmpMnt * cAppliCmpMnt::StdAlloc(const std::string & aNameXML)
{
    cCompareMNT aCmp = StdGetObjFromFile<cCompareMNT>
                       (
                            aNameXML,
                            "include/XML_GEN/ParamChantierPhotogram.xml",
                            "CompareMNT",
                            "CompareMNT"
                       );
    return new cAppliCmpMnt(aCmp);
}

Pt2di cAppliCmpMnt::Sz() const
{ 
  return mSz;
}

void cAppliCmpMnt::ShowDiff
     (
           const std::string & aMes,
           bool  isSigned,  // Visu + calcul de biais
           double aDynVisu
     )
{
   Im2D_Bits<1> aMasq = mCurZ->Masq();


   Symb_FNum  fD = mImDiff.in();

   double aS0,aS1,aS2;

   ELISE_COPY
   (
       select(aMasq.all_pts(),aMasq.in()),
       Virgule(fD,Square(fD),1),
       Virgule(sigma(aS1),sigma(aS2),sigma(aS0))
   );

   aS1 /= aS0;
   aS2 /= aS0;
   aS2 -= ElSquare(aS1);
   aS2 = sqrt(aS2);

   mCout <<   "          #   " << aMes;
   mCout << ": Moy " << aS1  ;
   if (isSigned)
       mCout << ", Ec2 " << aS2  ;
    
   mCout << "\n";

   Video_Win * aW = mCurTest->W();
   Fonc_Num aFonc = mImDiff.in() *aDynVisu;
   if (isSigned)
      aFonc = 128 + aFonc;
   if (aW)
   {
       ELISE_COPY
       (
           aW->all_pts(),
	   Max(0,Min(255,aFonc)),
	   aW->ogray()
       );

   }
}


void cAppliCmpMnt::DoOneCmp()
{
    mCout << "   *  "  <<  mCurRef->ShortName () << " (Ref) "
          << mCurTest->ShortName ()  << " (Test) "  
	  << "\n";
    if (mArg.EcartZ().IsInit())
    {
        mCurRef->EcartZ(mImDiff,*mCurTest);
	ShowDiff
	(
	    "Ecart Z",
	    true,   // Signe
            mArg.EcartZ().Val().DynVisu()      // Dynamique
	);
    }
    if (mArg.CorrelPente().IsInit())
    {
        mCurRef->CorrelPente(mImDiff,*mCurTest,mArg.CorrelPente().Val());
	ShowDiff
	(
	    "Correletion Pente",
	    true,   // Signe
            128      // Dynamique
	);
        // getchar();
    }
    //getchar();

}

void cAppliCmpMnt::DoAllCmp()
{

    // mFP = FopenNN(mArg.NameFileRes(),"w","cAppliCmpMnt::NameFileRes");

    for (int aKZ=0 ; aKZ<int(mZones.size()) ; aKZ++)
    {
        mCurZ = mZones[aKZ];
	mCout << "----  ["<< aKZ 
	      << "] ---[" << mCurZ->Nom() << "]-----------\n";
	mCout << "     \n";
	mCout << "    Valeurs Moyennes \n\n";
	for (int aKM=0 ; aKM<int(mMnts.size()) ; aKM++)
	{
            mMnts[aKM]->Load(*mZones[aKZ]);
	    if (mMnts[aKM]->IdRef())
	    {
                double  aZMoy,aPenteMoy;
	        mMnts[aKM]->CalcVMoy(aZMoy,aPenteMoy);
		mCout << "        [" << mMnts[aKM]->ShortName() << " ] "
		      << " ZMoy " << aZMoy
		      << " Pente " <<  aPenteMoy
		      << "\n";
	    }
	}
	if (mArg.VisuInter().Val()) 
	    getchar();
	Pt2di aSzZ = mZones[aKZ]->Box().sz();
	mImDiff = Im2D_REAL4(aSzZ.x,aSzZ.y);

	mCout << "     \n";
	mCout << "    Comparaisons \n\n";
	for (int aKM1=0 ; aKM1<int(mMnts.size()) ; aKM1++)
	{
            mCurRef= mMnts[aKM1];
	    for (int aKM2=0 ; aKM2<int(mMnts.size()) ; aKM2++)
	    {
               mCurTest= mMnts[aKM2];
	       if (mMnts[aKM1]->IdRef() >mMnts[aKM2]->IdRef())
               {
	           DoOneCmp();
               }
	    }
	}

    }
}





};



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

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


std::list<std::string> SplitStrings(const std::string & aCppStr);


/******************************************************/
/*                                                    */
/*              cElImplemBitmFont                     */
/*                                                    */
/******************************************************/

class cElImplemBitmFont : public cElBitmFont
{
    public :

      cElImplemBitmFont(Pt2di aSz,const std::string & aName);
      void AddChar(char,const char*);

   private :
      virtual Im2D_Bits<1> ImChar(char) ;


      std::map<char, Im2D_Bits<1> *> mDico;

      Pt2di mSz;
      std::string mName;
};

cElImplemBitmFont::cElImplemBitmFont(Pt2di aSz,const std::string & aName) :
   mSz   (aSz),
   mName (aName)
{
}

Im2D_Bits<1> cElImplemBitmFont::ImChar(char aChar) 
{
   Im2D_Bits<1> * aRes = mDico[aChar];
   if (aRes == 0)
   {
      std::cout << "Cannot Get Im for font " << mName 
                << " With Char [" << aChar << "]\n";
      ELISE_ASSERT(false,"Fonte probleme");
   }
   return *aRes;
}

void cElImplemBitmFont::AddChar(char anEntry,const char* aStr)
{
   // std::cout << "cElImplemBitmFont::AddChar " << mSz << " " << strlen(aStr) << " " << anEntry << "\n";
   if (strlen(aStr)!=(size_t)(mSz.x*mSz.y))
   {
       std::cout << "FOR CAR=[" << anEntry << "] ASCII=" << int(anEntry) << "\n";
       ELISE_ASSERT(false,"cElImplemBitmFont::AddChar bad size");
   }
   mDico[anEntry] = new Im2D_Bits<1> (mSz.x,mSz.y);
   Im2D_Bits<1>  * aVal = mDico[anEntry];

   for (int anY=0; anY<mSz.y ; anY++)
   {
       for (int anX=0; anX<mSz.x ; anX++)
       {
           aVal->set(anX,anY,*(aStr++) == '#');
       }
   }
}


Im2D_Bits<1> cElBitmFont::BasicImageString(const std::string & aStr,int aSpace)
{
   bool  isHor = (aSpace>=0);
   aSpace = std::abs(aSpace);

   if (aStr=="")
   {
      return  ImChar(' ');
   }
   Pt2di aSzRes(0,0);
   Pt2di aSzIm (0,0);

   for (const char * aC = aStr.c_str(); (*aC) ; aC++) 
   {
      Im2D_Bits<1> anIm = ImChar(*aC);
      aSzIm = anIm.sz();
      if (aSzRes.x==0)
      {
           aSzRes = aSzIm;
      }
      else
      {
           if (isHor)
           {
               aSzRes.x += aSzIm.x + aSpace;
               ELISE_ASSERT(aSzRes.y==aSzIm.y,"ImageString Height variable for font");
           }
           else
           {
               aSzRes.y += aSzIm.y + aSpace;
               ELISE_ASSERT(aSzRes.x==aSzIm.x,"ImageString Width variable for font");
           }
      }
   }

   Im2D_Bits<1> aRes(aSzRes.x,aSzRes.y,0);
   Pt2di anOffset(0,0);

   for (const char * aC = aStr.c_str(); (*aC) ; aC++) 
   {
      Im2D_Bits<1> anIm = ImChar(*aC);
      Pt2di aSzIm = anIm.sz();
      ELISE_COPY
      (
           rectangle(anOffset,aSzIm+anOffset), 
           trans( anIm.in(),-anOffset),
           aRes.out()
      );
      if (isHor)
         anOffset.x += aSzIm.x + aSpace;
      else
         anOffset.y += aSzIm.y + aSpace;
   }

   return aRes;
}

Im2D_Bits<1> cElBitmFont::MultiLineImageString(const std::string & aStrInit,Pt2di  aSpace,Pt2di aRab,int Centering)
{
     std::list<std::string> aLStr =  SplitStrings(aStrInit);

     std::list<Im2D_Bits<1> > aLIm;
  
     int aXMax=0;
     int aSomY = 0;

     std::list<Im2D_Bits<1> > aLB;
     for (std::list<std::string>::const_iterator itS=aLStr.begin() ;itS!=aLStr.end() ; itS++)
     {
         Im2D_Bits<1>  anIm = BasicImageString(*itS,aSpace.x);
         Pt2di aSz = anIm.sz();
         aXMax = ElMax(aXMax,aSz.x);
         aSomY += aSz.y;
         aLB.push_back(anIm);
     }
     aSomY += aSpace.y*(int)(aLStr.size()-1);
     if (aRab.x<0)  // Alors Rab est une taille totale
     {
        aRab = - aRab;
        aRab = (aRab- Pt2di(aXMax,aSomY))/2;
     }

     Pt2di aSzRes = (aRab*2) +Pt2di(aXMax,aSomY);

     Im2D_Bits<1>  aRes(aSzRes.x,aSzRes.y);

     Pt2di aP0(0,aRab.y);

     for (std::list<Im2D_Bits<1> >::const_iterator itB=aLB.begin() ;itB!=aLB.end() ; itB++)
     {
            
         Im2D_Bits<1>  anIm = *itB;
         Pt2di aSz =  anIm.sz();

         aP0.x = aRab.x; // cas -1
         if (Centering==0)
            aP0.x = (aSzRes.x-aSz.x) /2;
         else if (Centering>0)
            aP0.x = (aSzRes.x-aSz.x) - aRab.x;

         ELISE_COPY
         (
              rectangle(aP0,aP0+aSz),
              trans(anIm.in(),-aP0),
              aRes.out()
         );
         aP0.y += aSz.y + aSpace.y;
     }

     return aRes;
}

Im2D_Bits<1> MMStrIcone(const std::string & aName)
{
    cElBitmFont & aFont = cElBitmFont::BasicFont_10x8() ;

    return aFont.MultiLineImageString(aName,Pt2di(0,1),Pt2di(5,5),0);
}

Im2D_Bits<1> MMStrIcone(const std::string & aName,const Pt2di & aSz)
{
    cElBitmFont & aFont = cElBitmFont::BasicFont_10x8() ;

    return aFont.MultiLineImageString(aName,Pt2di(0,1),-aSz,0);
}

/******************************************************/
/*                                                    */
/*              cElBitmFont                           */
/*                                                    */
/******************************************************/


/*
bool IsBackSlashMetaChar(const char * aStr,const char aVal)
{
   return (aStr[0]=='\\') && (aStr[1]==aVal);
}
*/

std::list<std::string> SplitStrings(const std::string & aCppStr)
{
   std::list<std::string> aRes;
   const char *  aStrInit = aCppStr.c_str();
   const char * aStr0 = aStrInit;
   bool Cont = (*aStr0!=0);

   while (Cont)
   {
        const char * aStr1 = aStr0;
        while ((*aStr1!=0) &&  (*aStr1 !='\n'))
           aStr1++;
        aRes.push_back(std::string(aStr0,aStr1));

        if (*aStr1==0)
          Cont= false;
        else
          aStr0 = aStr1+1;
   }

   return aRes;
}

void TestSplitStrings(const std::string & aCppStr)
{

   std::list<std::string> aRes = SplitStrings(aCppStr);

   std::cout << "**** IN={["<< aCppStr << "]}\n";
   for (std::list<std::string>::const_iterator itS=aRes.begin() ; itS!=aRes.end() ; itS++)
   {
       std::cout << "   ["<< *itS << "]\n";
   }
   std::cout << "************************** \n";
}

cElBitmFont::~cElBitmFont()
{
}

cElImplemBitmFont * cElBitmFont::theFont_10x8 = 0;
cElImplemBitmFont * cElBitmFont::theFontCodedTarget = 0;

cElBitmFont & cElBitmFont::FontCodedTarget()
{
  if (theFontCodedTarget !=0)
     return *theFontCodedTarget;

   theFontCodedTarget = new cElImplemBitmFont(Pt2di(11,11),"DCT");
 

   theFontCodedTarget->AddChar
   (
        '0',
        "..####....."   
        ".#....#...."
        ".#....#...."
        ".#....#...."
        ".#....#...."
        ".#....#...."
        ".#....#...."
        "..####....."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '1',
        "....#......"
        "...##......"
        "..#.#......"
        ".#..#......"
        "....#......"
        "....#......"
        "....#......"
        "....#......"
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '2',
        ".###......."
        "#...#......"
        "#....#....."
        "....#......"
        "...#......."
        "..#........"
        "######....."
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '3',
        "..##......."
        ".#..#......"
        "....#......"
        "..##......."
        ".....#....."
        ".#...#....."
        "..###......"
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '4',
        "....#......"
        "...#......."
        "..#........"
        ".#........."
        "#...#......"
        "######....."
        "....#......"
        "....#......"
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '5',
        ".#####....."
        ".#........."
        "#.........."
        "####......."
        "....#......"
        ".....#....."
        ".....#....."
        "#####......"
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '6',
        "...###....."
        "..#...#...."
        " #........."
        ".####......"
        "#.....#...."
        "#.....#...."
        ".#####....."
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '7',
        ".#####....."
        ".....#....."
        "....#......"
        "..####....."
        "...#......."
        "...#......."
        "..#........"
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '8',
        "...###....."
        "..#...#...."
        "..#...#...."
        "...###....."
        "..#...#...."
        ".#.....#..."
        ".#....#...."
        "..####....."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        '9',
        "..#####...."
        ".#.....#..."
        ".#....##..."
        "..####.#..."
        ".......#..."
        ".....#....."
        "..###......"
        "..........."
        "..........."
        "..........."
        "..........."
   );


   theFontCodedTarget->AddChar
   (
        'A',
        "...###....."
        "..#...#...."
        ".#.....#..."
        ".#######..."
        ".#.....#..."
        ".#.....#..."
        ".#.....#..."
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        'B',
        ".####......"
        ".#...#....."
        ".#..#......"
        ".######...."
        ".#.....#..."
        ".#.....#..."
        ".######...."
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        'C',
        "...####...."
        ". #...#...."
        ".#........."
        ".#........."
        "#.........."
        "#....#....."
        ".####......"
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        'D',
        "..####....."
        "..#...#...."
        "..#....#..."
        ".#....#...."
        ".#....#...."
        ".#..##....."
        "####......."
        "..........."
        "..........."
        "..........."
        "..........."
   );

   theFontCodedTarget->AddChar
   (
        'E',
        "..#####...."
        "..#........"
        ".#........."
        ".###......."
        ".#........."
        "#.........."
        "#####......"
        "..........."
        "..........."
        "..........."
        "..........."
   );
   theFontCodedTarget->AddChar
   (
        'F',
        "..######..."
        "..#........"
        "..#........"
        "..###......"
        ".#........."
        ".#........."
        ".#........."
        "..........."
        "..........."
        "..........."
        "..........."
   );

   return *theFontCodedTarget;
}


cElBitmFont & cElBitmFont::BasicFont_10x8()
{
  if (theFont_10x8 !=0)
     return *theFont_10x8;

   theFont_10x8 = new cElImplemBitmFont(Pt2di(10,10),"El_10x8");
 
   theFont_10x8->AddChar
   (
        '_',
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."
        "..#######."

        ".........."
        ".........."
   );
   theFont_10x8->AddChar
   (
        ' ',
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '.',
        ".........."   
        ".........."
        ".........."
        ".........."
        ".........."
        ".........."
        "....##...."
        "....##...."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '0',
        "...####..."   
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "...####..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '1',
        "......#..."
        ".....##..."
        "....#.#..."
        "...#..#..."
        "......#..."
        "......#..."
        "......#..."
        "....####.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '2',
        "...###...."
        "..#...#..."
        "..#....#.."
        "......#..."
        ".....#...."
        "....#....."
        "...#......"
        "..######.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '3',
        "....##...."
        "...#..#..."
        "......#..."
        "....##...."
        "......#..."
        ".......#.."
        "...#...#.."
        "....###..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '4',
        "......#..."
        ".....#...."
        "....#....."
        "...#......"
        "..#...#..."
        "..######.."
        "......#..."
        "......#..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '5',
        "..#####..."
        "..#......."
        "..#......."
        "..####...."
        "......#..."
        ".......#.."
        ".......#.."
        "..#####..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '6',
        ".....###.."
        "....#....."
        ".. #......"
        "..#......."
        "..#.####.."
        "..##....#."
        "..#.....#."
        "...#####.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '7',
        "...#####.."
        ".......#.."
        "......#..."
        "....####.."
        ".....#...."
        ".....#...."
        "....#....."
        "....#....."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '8',
        "...####..."
        "..#....#.."
        "..#....#.."
        "...####..."
        "..#....#.."
        ".#......#."
        ".#......#."
        "..######.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        '9',
        "..#####..."
        ".#.....#.."
        ".#....##.."
        "..####.#.."
        ".......#.."
        "......#..."
        ".....#...."
        "..###....."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
   (
        'A',
        "...####..."
        "..#....#.."
        ".#......#."
        ".#......#."
        ".########."
        ".#......#."
        ".#......#."
        ".#......#."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'B',
        ".#####...."
        ".#....#..."
        ".#....#..."
        ".#####...."
        ".#.....#.."
        ".#......#."
        ".#.....#.."
        ".######..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'C',
        "...####..."
        "..#....#.."
        "..#......."
        "..#......."
        "..#......."
        "..#......."
        "..#....#.."
        "...####..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'D',
        "..####...."
        "..#...#..."
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "..#...#..."
        "..####...."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'E',
        "..######.."
        "..#......."
        "..#......."
        "..#####..."
        "..#......."
        "..#......."
        "..#......."
        "..######.."

        ".........."
        ".........."
   );
   theFont_10x8->AddChar
   (
        'F',
        "..######.."
        "..#......."
        "..#......."
        "..#####..."
        "..#......."
        "..#......."
        "..#......."
        "..#......."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'G',
        "...####..."
        "..#....#.."
        "..#......."
        "..#......."
        "..#.. ###."
        "..#....#.."
        "..#....#.."
        "...####..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'H',
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "..######.."
        "..#....#.."
        "..#....#.."
        "..#....#.."
        "..#....#.."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
   (
        'I',
        "..#######."
        ".....#...."
        ".....#...."
        ".....#...."
        ".....#...."
        ".....#...."
        ".....#...."
        "..#######."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'J',
        "..#######."
        ".......#.."
        ".......#.."
        ".......#.."
        ".......#.."
        ".......#.."
        "..#...#..."
        "...###...."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'K',
        "..#....#.."
        "..#...#..."
        "..#..#...."
        "..##......"
        "..#..#...."
        "..#...#..."
        "..#....#.."
        "..#.....#."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'L',
        "..#......."
        "..#......."
        "..#......."
        "..#......."
        "..#......."
        "..#......."
        "..#......."
        "..#######."

        ".........."
        ".........."
   );




   theFont_10x8->AddChar
   (
        'M',
        "..#.....#."
        "..##...##."
        "..#.#.#.#."
        "..#..#..#."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
	   (
	   'N',
	   "..#.....#."
	   "..##....#."
	   "..#.#...#."
	   "..#..#..#."
	   "..#...#.#."
	   "..#....##."
	   "..#.....#."
	   "..#.....#."

	   ".........."
	   ".........."
	   );

   theFont_10x8->AddChar
	   (
	   'O',
	   "...####..."
	   "..#....#.."
	   ".#......#."
	   ".#......#."
	   ".#......#."
	   ".#......#."
	   "..#....#.."
	   "...####..."
	   ".........."
	   ".........."
	   );

   theFont_10x8->AddChar
   (
        'P',
        "..####...."
        "..#...##.."
        "..#....#.."
        "..#...##.."
        "..####...."
        "..#......."
        "..#......."
        "..#......."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'Q',
        "...####..."
        "..#....#.."
        ".#......#."
        ".#......#."
        ".#......#."
        ".#......#."
        "..#.#..#.."
        "...####..."
        ".....#...."
        "......#..."
   );


   theFont_10x8->AddChar
   (
        'R',
        "..#####..."
        "..#....#.."
        "..#....#.."
        "..#####..."
        "..#....#.."
        "..#.....#."
        "..#.....#."
        "..#.....#."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
   (
        'S',
        "...####..."
        "..#....#.."
        "..#......."
        "...#####.."
        "........#."
        "........#."
        "..#.....#."
        "...#####.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'T',
        ".########."
        ".....#...."
        ".....#...."
        ".....#...."
        ".....#...."
        ".....#...."
        ".....#...."
        ".....#...."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'U',
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#....##."
        "...####.#."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'V',
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "...#...#.."
        "...#...#.."
        "...#...#.."
        "....#.#..."
        ".....#...."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'W',
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#..#..#."
        "..#.#.#.#."
        "..##...##."
        "..#.....#."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'X',
        "..#.....#."
        "...#...#.."
        "....#.#..."
        ".....#...."
        "....#.#..."
        "...#...#.."
        "..#.....#."
        "..#.....#."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
	   (
	   'Y',
	   "..#.....#."
	   "...#...#.."
	   "....#.#..."
	   ".....#...."
	   ".....#...."
	   ".....#...."
	   ".....#...."
	   ".....#...."

	   ".........."
	   ".........."
	   );

   theFont_10x8->AddChar
   (
        'Z',
        ".########."
        ".#.....#.."
        "......#..."
        ".....#...."
        "....#....."
        "...#......"
        "..#.....#."
        ".########."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
   (
        'a',
        ".........."
        ".........."
        "...####..."
        ".......#.."
        "...#####.."
        "..#....#.."
        "..#...##.."
        "...###.#.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
	   (
	   'b',
	   ".........."
	   ".........."
	   "..#......."
	   "..#......."
	   "..#####..."
	   "..#....#.."
	   "..#....#.."
	   "..#####..."

	   ".........."
	   ".........."
	   );

   theFont_10x8->AddChar
   (
        'c',
        ".........."
        ".........."
        ".........."
        "...####..."
        "..#....#.."
        "..#......."
        "..#....#.."
        "...####..."

        ".........."
        ".........."
   );
   theFont_10x8->AddChar
   (
        'd',
        ".........."
        ".........."
        ".......#.."
        ".......#.."
        "...#####.."
        "..#....#.."
        "..#....#.."
        "...#####.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'e',
        ".........."
        ".........."
        "...####..."
        "..#....#.."
        "..#....#.."
        "..#####..."
        "..#......."
        "...####..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'f',
        ".........."
        ".....###.."
        "....#....."
        "...#......"
        "..####...."
        "...#......"
        "...#......"
        "...#......"

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'g',
        ".........."
        ".........."
        ".........."
        "...###.#.."
        "..#...##.."
        "..#....#.."
        "..#....#.."
        "...#####.."

        ".......#.."
        "...####..."
   );

   theFont_10x8->AddChar
   (
        'h',
        ".........."
        "..#......."
        "..#......."
        "..#......."
        "..#####..."
        "..#....#.."
        "..#....#.."
        "..#....#.."

        ".........."
        ".........."
   );





   theFont_10x8->AddChar
   (
        'i',
        ".........."
        "...#......"
        ".........."
        "..##......"
        "...#......"
        "...#......"
        "....#....."
        ".....##..."

        ".........."
        ".........."
   );
   theFont_10x8->AddChar
   (
        'j',
        ".........."
        "......#..."
        ".........."
        "......#..."
        "......#..."
        "......#..."
        "......#..."
        "......#..."

        ".....#...."
        "..###....."
   );
   theFont_10x8->AddChar
   (
        'k',
        ".........."
        "..#......."
        "..#......."
        "..#...#..."
        "..#..#...."
        "..###....."
        "..#..#...."
        "..#...#..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'l',
        "..##......"
        "...#......"
        "...#......"
        "...#......"
        "...#......"
        "...#......"
        "....#..#.."
        ".....##..."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
   (
        'm',
        ".........."
        ".........."
        "..#.#.##.."
        "..##.#..#."
        "..#..#..#."
        "..#..#..#."
        "..#..#..#."
        "..#..#..#."

        ".........."
        ".........."
   );
   theFont_10x8->AddChar
   (
        'n',
        ".........."
        ".........."
        "..#.####.."
        "..##....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'o',
        ".........."
        ".........."
        "....###..."
        "...#...#.."
        "..#.....#."
        "..#.....#."
        "...#...#.."
        "....###..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'p',
        ".........."
        ".........."
        "..####...."
        "..#...#..."
        "..#....#.."
        "..#....#.."
        "..#...#..."
        "..####...."
        "..#......."
        "..#......."
   );


   theFont_10x8->AddChar
   (
        'q',
        ".........."
        ".........."
        "...###.#.."
        "..#...##.."
        "..#....#.."
        "..#....#.."
        "..#...##.."
        "...###.#.."
        ".......#.."
        ".......#.."
   );

   theFont_10x8->AddChar
   (
        'r',
        ".........."
        ".........."
        "..#.####.."
        "..##......"
        "..#......."
        "..#......."
        "..#......."
        "..#......."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
   (
        's',
        ".........."
        ".........."
        "...####..."
        "..#......."
        "...####..."
        ".......#.."
        "..#....#.."
        "...####..."

        ".........."
        ".........."
   );


   theFont_10x8->AddChar
   (
        't',
        ".........."
        ".........."
        "...#......"
        "..####...."
        "...#......"
        "...#......"
        "...#...#.."
        "....###..."

        ".........."
        ".........."
   );
   theFont_10x8->AddChar
   (
        'u',
        ".........."
        ".........."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#.....#."
        "..#....##."
        "...####.#."

        ".........."
        ".........."
   );
   theFont_10x8->AddChar
   (
        'v',
        ".........."
        ".........."
        "..#.....#."
        "..#.....#."
        "...#...#.."
        "...#...#.."
        "....#.#..."
        ".....#...."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
        'w',
        ".........."
        ".........."
        "#.......#."
        "#.......#."
        ".#..#..#.."
        ".#..#..#.."
        "..##.##..."
        "..#...#..."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
   (
		'x',
	".........."
        ".........."
        ".........."
        "..#....#.."
        "...#..#..."
        "....##...."
        "...#..#..."
        "..#....#.."

        ".........."
        ".........."
   );

   theFont_10x8->AddChar
	   (
	   'y',
	   ".........."
	   ".........."
	   ".........."
	   "..#.....#."
	   "...#...#.."
	   "....###..."
	   ".....#...."
	   ".....#...."

	   ".........."
	   ".........."
	   );


   theFont_10x8->AddChar
   (
        'z',
        ".........."
        ".........."
        "..######.."
        "......#..."
        ".....#...."
        "....#....."
        "...#......"
        "..######.."

        ".........."
        ".........."
   );

   return *theFont_10x8;
}

/*

static char * c7 =
static char * c8 =
static char * c9 =
static char * cA =
"....##...."
"...#..#..."
"..#....#.."
"..######.."
"..#....#.."
"..#....#.."
".#......#."
".#......#.";
static char * cB =
"..#####..."
"...#...#.."
"...#...#.."
"...#####.."
"...#....#."
"..#.....#."
"..#....#.."
".######...";
static char * cC =
"...####..."
"..#....#.."
"..#......."
"..#......."
"..#......."
"..#......."
"..#....#.."
"...####...";



static char * cX =
".........."
".........."
".........."
".........."
".........."
".........."
".........."
"..........";
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

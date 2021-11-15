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


/*********************************************************************/
/*                                                                   */
/*                  Data_BMP_File                                    */
/*                                                                   */
/*********************************************************************/

void pgm_eol(ELISE_fp fp)
{
    while (fp.read_U_INT1() != '\n');
}

INT pgm_get_int(ELISE_fp fp,U_INT1 &c)
{
    while
    (
          ((c = fp.read_U_INT1()) < '0')
       || (c > '9')
    )
    { 
        if (c=='#')
           pgm_eol(fp);
    }

    INT res = c-'0';
    while
    (
          (((c = fp.read_U_INT1()) >= '0') && (c <= '9')) || (c=='.') || (c=='-')
    )
       res = 10*res+c-'0';
    return res;
}

Elise_File_Im  Elise_File_Im::pnm(const char * name)
{
     ELISE_fp fp;

     Tjs_El_User.ElAssert 
     (
          fp.ropen(name,true),
          EEM0 << "Elise_File_Im::pnm : can't open the file \n"
               << "|       File name : " << name
     );

     int aCar = fp.read_U_INT1();
     ASSERT_TJS_USER((aCar == 'P'),"bad magic pnm");
     INT nb_can = -1234;
     INT kind_pnm = fp.read_U_INT1();
     bool isFloatIm = false;
     switch (kind_pnm)
     {
           case '4' : nb_can = 1; break;  // 1 Bits  images
           case '5' : nb_can = 1; break;  // 
           case '6' : nb_can = 3; break;
           case 'f' : nb_can = 1; isFloatIm = true;break;
           case 'F' : nb_can = 3; isFloatIm = true;break;
           default :  elise_fatal_error("bad magic pbm",__FILE__,__LINE__);
     }
     

     U_INT1 last_c;
     INT sz[2];
     sz[0]  = pgm_get_int(fp,last_c);
     sz[1]  = pgm_get_int(fp,last_c);

     int aNbMaxVal = 1;
     if (kind_pnm!='4')  // no max val for 1-bits images
     {
         aNbMaxVal = pgm_get_int(fp,last_c);
     }

     if (last_c != '\n')
         pgm_eol(fp);

     tFileOffset offs = fp.tell();
     fp.close();

     GenIm::type_el  aType = GenIm::u_int1;
     if ( kind_pnm==4)
         aType = GenIm::bits1_msbf;
     else if (isFloatIm )
         aType = GenIm::real4;
     else if (aNbMaxVal>=256)
         aType = GenIm::u_int2;
        

     return Elise_File_Im
            (
                 name,
                 2,
                 sz,
		 aType,
                 // (kind_pnm=='4') ? GenIm::bits1_msbf : GenIm::u_int1,
                 nb_can,
                 offs
            );
}


Elise_File_Im Elise_File_Im::pnm 
(
          const char *    name,
          char **         comment,
          Pt2di           sz,
          GenIm::type_el  type,
          INT    dim_out,
          INT  mode_pnm
)
{
/*  Par homogeneite avec le reste on ecrase les fichiers.
    if (ELISE_fp::exist_file(name))
       return Elise_File_Im::pnm(name);
*/

    ELISE_fp   fp(name,ELISE_fp::WRITE);

    char buf[200];
    sprintf(buf,"P%d\n",mode_pnm);
    fp.str_write(buf);
    fp.str_write("# SOFTWARE : ELISE (by Marc PIERROT DESEILLIGNY)\n");
    if (comment)
       for ( ; *comment ; comment++)
       {
            fp.str_write("#");
            fp.str_write(*comment);
            fp.str_write("\n");
       }

    sprintf(buf,"%d %d\n",sz.x,sz.y);
    fp.str_write(buf);
    
    INT vmax = (1<<nbb_type_num(type))-1;
    if (vmax != 1)
    {
         sprintf(buf,"%d\n",vmax);
         fp.str_write(buf);
    }
    tFileOffset offs0 = fp.tell();
    fp.close();

    INT txy[2];
    sz.to_tab(txy);

    return Elise_File_Im
           (
               name,
               2,
               txy,
               type,
               dim_out,
               offs0,
               -1,
               true
           );
}


Elise_File_Im Elise_File_Im::pbm 
              (
                    const char * name,
                    Pt2di  sz,
                    char **  comment
              )
{
     return Elise_File_Im::pnm 
            (
                 name,
                 comment,
                 sz,
                 GenIm::bits1_msbf,
                 1,
                 4
             );
}

Elise_File_Im Elise_File_Im::pgm 
              (
                    const char * name,
                    Pt2di  sz,
                    char **  comment
              )
{
     return Elise_File_Im::pnm 
            (
                 name,
                 comment,
                 sz,
                 GenIm::u_int1,
                 1,
                 5
             );
}

Elise_File_Im Elise_File_Im::ppm 
              (
                    const char * name,
                    Pt2di  sz,
                    char **  comment
              )
{
     return Elise_File_Im::pnm 
            (
                 name,
                 comment,
                 sz,
                 GenIm::u_int1,
                 3,
                 6
             );
}


/*********************************************************************/
/*                                                                   */
/*                  sun raster file                                  */
/*                                                                   */
/*********************************************************************/


Elise_Tiled_File_Im_2D  Elise_Tiled_File_Im_2D ::sun_raster(const char * name)
{
	 ELISE_fp  fp(name,ELISE_fp::READ);

	 INT Magic = fp.msb_read_INT4();
	 INT tx = fp.msb_read_INT4();
	 INT ty = fp.msb_read_INT4();
	 INT Bitspp = fp.msb_read_INT4();
	 /* INT length = */ fp.msb_read_INT4();
	 INT type = fp.msb_read_INT4();
	 fp.msb_read_INT4(); //  ColorMap
	 INT NbByteCMap = fp.msb_read_INT4();


     Tjs_El_User.ElAssert
	 (
		Magic == 0x59a66a95,
		EEM0 << "Bad Magig number for Sun Raster File " << name
	 );

     Tjs_El_User.ElAssert
	 (
		(type == 0) || (type == 1),
		EEM0 << "Do not handle compression for Sun Raster File " << name
	);

	INT depth  =  (Bitspp==1) ? 1  : 8;
	INT nb_chan = Bitspp / depth;


	fp.close();

	return Elise_Tiled_File_Im_2D
		   (
		        name,
                Pt2di(tx,ty),
                type_im(true,depth,false,true),
                nb_chan,
                Pt2di(tx,ty),
				true,
				true,
				32+NbByteCMap,
				false,
                MSBF_PROCESSOR()
		   );
}

bool IsKnowImagePostFix(const std::string & aPostMix)
{
   std::string aPost = StrToLower(aPostMix);

   switch (aPost[0])
   {
        case 'a' :
             return    (aPost=="arw") 
             ;

        case 'c' :
             return    (aPost=="cr2") 
             ;

        case 'j' :
             return    (aPost=="jpg") 
                    || (aPost=="jpeg")
             ;

        case 'n' :
             return    (aPost=="nef") 
             ;

        case 'p' :
             return    (aPost=="pef") 
                    || (aPost=="pbm")
                    || (aPost=="pgm")
                    || (aPost=="ppm")
                    || (aPost=="pfm")
             ;

        case 't' :
             return   (aPost=="tif") 
                   || (aPost=="tiff") 
             ;
   }

   return false;
}

std::string * TheGlobNameRaw = 0;

cSpecifFormatRaw * GetSFRFromString(const std::string & aNameHdr)
{
  static std::map<std::string,cSpecifFormatRaw *> aMapRes;

  std::map<std::string,cSpecifFormatRaw *>::iterator anIt = aMapRes.find(aNameHdr);

  if (anIt != aMapRes.end())  return anIt->second;


  cSpecifFormatRaw * aRes = 0;
  if ( ! IsKnowImagePostFix(aNameHdr))
  {
      std::string aDir,aNameSeul;
      SplitDirAndFile(aDir,aNameSeul,aNameHdr);

      static cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::Glob();
      if (anICNM==0) anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);


      std::string aNameFile = aNameHdr;
      if (StdPostfix(aNameHdr)!="xml")
      {
           aNameFile = aDir + anICNM->Assoc1To1("NKS-Assoc-SpecifRaw",aNameSeul,true);
      }

      if (TheGlobNameRaw!=0)  
         aNameFile = *TheGlobNameRaw;


      if (ELISE_fp::exist_file(aNameFile))
      {
          aRes = OptionalGetObjFromFile_WithLC<cSpecifFormatRaw>
                 (
                   0,0,
                   aNameFile,
                   StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                   "SpecifFormatRaw",
                   "SpecifFormatRaw"
                 );
          if (aRes && (! aRes->NameFile().IsInit()))
            aRes->NameFile().SetVal(aNameHdr);
      // aRes->
      }
  }



  aMapRes[aNameHdr] = aRes;
  return aRes;
}
/*
*/


Elise_Tiled_File_Im_2D 
    Elise_Tiled_File_Im_2D::XML(const std::string & aNameHdr)
{
/*
   cSpecifFormatRaw aSpec = 
   StdGetObjFromFile<cSpecifFormatRaw>
   (
       aNameHdr,
        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
       "SpecifFormatRaw",
       "SpecifFormatRaw"
   );
*/
   cSpecifFormatRaw & aSpec = *(GetSFRFromString(aNameHdr));

   return Elise_Tiled_File_Im_2D
          (
	       aSpec.NameFile().Val().c_str(),  // Name
	       aSpec.Sz(),                      // sz
	       type_im                          // type
	       (
	            aSpec.IntegerType(),
	            aSpec.NbBitsParPixel(),
		    aSpec.SignedType(),
		    false
	       ),
	       1,                               // dim_out,
	       aSpec.Sz(),                      // sz_tiles,
	       DefCLT,                          // clip_last_tile,
	       DefChunk,                                // chunk,
	       aSpec.Offset().ValWithDef(DefOffset0),   // offset_0
	       DefCreate,                               // create
	       aSpec.MSBF() == MSBF_PROCESSOR()         // byte_ordered
	  );
}

Elise_Tiled_File_Im_2D 
    Elise_Tiled_File_Im_2D::HDR(const std::string & aNameHdr)
{
    std::string aPref = StdPrefix(aNameHdr);
    int          aTx,aTy,aNbBits;
    std::string  aNameLayout,aByteOrder,aSigned;
    aByteOrder = "I";
    aSigned = "false";
    // aNameLayout= "BIL";
    aNbBits = 8;
    HdrInitArgsFromFile
    (
          aNameHdr,
          LArgMain() 
                     << EAM(aTx,"NCOLS",false)
                     << EAM(aTy,"NROWS",false)
                     << EAM(aNbBits,"NBITS",true)
                     << EAM(aSigned,"SIGNED",true)
                     << EAM(aNameLayout,"LAYOUT",false)
                     << EAM(aByteOrder,"BYTEORDER",true)
    );


    bool IsIntegral=true;
    bool IsSigned= aSigned != "false";
    // IsSigned = true; std::cout << "SIGNNNEEEDDD \n\n";
    GenIm::type_el aType =  type_im(IsIntegral,aNbBits,IsSigned,true);

    std::string aNameData = aPref + "." + aNameLayout;

    std::cout  << "ByO=" << aByteOrder << " " << MSBF_PROCESSOR() << "\n";
    bool IsByteOrdered  = MSBF_PROCESSOR() ?
                          (aByteOrder=="I") :
			  (aByteOrder!="I") ;
    IsByteOrdered = DefByteOrdered;
    return Elise_Tiled_File_Im_2D
           (
                aNameData.c_str(),
                Pt2di(aTx,aTy),
                aType,
                1,
                Pt2di(aTx,aTy),
                DefCLT,     
                DefChunk,    
                DefOffset0,
                DefCreate,
                IsByteOrdered // DefByteOrdered
           );
    
}

Elise_Tiled_File_Im_2D Elise_Tiled_File_Im_2D::Saphir(const char * name_file,const char * name_header)
{

    ElSTDNS vector<int> origine;
    ElSTDNS vector<int> dimensions;
    int         header;
    int         tail;
    ElSTDNS vector<int> block;
    ElSTDNS string      block_truncated;
    int         size;
    ElSTDNS string      type;
    ElSTDNS string      format;
    int         line_head;
    int         line_tail;
    int         block_head;
    int         block_tail;
    bool         MSBF = MSBF_PROCESSOR();


    SphInitArgs
    (
          name_header,
          LArgMain() << EAM(origine,"origine",false)
                     << EAM(dimensions,"dimensions",false)
                     << EAM(header,"header",false)
                     << EAM(tail,"tail",false)
                     << EAM(block,"block",false)
                     << EAM(block_truncated,"block-truncated",false)
                     << EAM(size,"size",false)
                     << EAM(type,"type",false)
                     << EAM(format,"format",false)
                     << EAM(line_head,"line_head",false)
                     << EAM(block_tail,"block_tail",false)
                     << EAM(block_head,"block_head",false)
                     << EAM(line_tail,"line_tail",false)
                     << EAM(MSBF,"MSBF",true)
    );

    ELISE_ASSERT
    (
        (origine.size()==2) && (dimensions.size()==2) && (block.size()==2),
        "Bad Arg Number for a point in  Elise_Tiled_File_Im_2D::Saphir"
    );

    ELISE_ASSERT
    (
        (line_head==0) &&(line_tail==0) &&(block_head==0) &&(block_tail==0),
        "line/block-head/tail != 0 in   Elise_Tiled_File_Im_2D::Saphir"
    );


    bool IsSigned = true;
    bool IsIntegral = false;


    if (type=="signed")
    {
        IsSigned = true;
        IsIntegral = true;
    }
    else if (type=="unsigned")
    {
        IsSigned = false;
        IsIntegral = true;
    }
    else if (type=="float")
    {
        IsSigned = true;
        IsIntegral = false;
    }
    else
    {
         ELISE_ASSERT(false,"Unknown type in Elise_Tiled_File_Im_2D::Saphir");
    }

     bool clip_last_tile = (block_truncated == "TRUE");

    GenIm::type_el aType =  type_im(IsIntegral,size,IsSigned,true);

    cout << "MSBF =[" << MSBF_PROCESSOR() << "]\n";
/*
    cout << "FORMAT =[" << format.c_str() << "]\n";
    cout << "dimensions " << dimensions << "\n";
    cout << "block " << block << "\n";
    cout  << "Type : " << aType << " " << GenIm::u_int1 << "\n";
*/


    return Elise_Tiled_File_Im_2D
           (
                name_file,
                Pt2di(dimensions[0],dimensions[1]),
                aType,
                1,
                Pt2di(block[0],block[1]),
                clip_last_tile,
                true,
                header,
                false,
                MSBF ==  MSBF_PROCESSOR()
           );

}

const bool Elise_Tiled_File_Im_2D::DefCLT         = false;
const bool Elise_Tiled_File_Im_2D::DefChunk       = true;
const int  Elise_Tiled_File_Im_2D::DefOffset0     = 0;
const bool Elise_Tiled_File_Im_2D::DefCreate      = false;
const bool Elise_Tiled_File_Im_2D::DefByteOrdered = true;
 


void MakeFileThomVide
     (
           const std::string & aNameVide,
	   const std::string& aNamePlein
     )
{
  if (ELISE_fp::exist_file(aNameVide))
      return;

  FILE * aFpIn = ElFopen(aNamePlein.c_str(),"r");
  if (aFpIn==0)
     cout << "File=[" << aNamePlein << "]\n";
  ELISE_ASSERT(aFpIn!=0,"MakeFileThomVide,  In");
  FILE * aFpOut = ElFopen(aNameVide.c_str(),"w");
  ELISE_ASSERT(aFpOut!=0,"MakeFileThomVide,  Out");

  if (! IsThomFile(aNamePlein))
  {
      Tiff_Im aTif = Tiff_Im::BasicConvStd(aNamePlein);
      Pt2di aSz = aTif.sz();
      int   aNbB =  aTif.bitpp();
      ELISE_ASSERT(aNbB%8==0,"MakeFileThomVide from Tiff_Im::StdConv");
      fprintf(aFpOut,"C ENTETE\n");
      fprintf(aFpOut,"C !!!!!!  ACHTUNG !!!!!!!!!\n");
      fprintf(aFpOut,"C !!!!!!  Fichier Thom \"bidon\" !!!!!!!!!\n");
      fprintf(aFpOut,"C !!!!!!  Seuls NBCOL et NBLIG sont OK  !!!!!!!!!\n");

      fprintf(aFpOut,"S BIDON=1\n");
      fprintf(aFpOut,"S NBCOL=%d\n",aSz.x);
      fprintf(aFpOut,"S NBLIG=%d\n",aSz.y);
      fprintf(aFpOut,"A NOM=%s\n",aNamePlein.c_str());
      fprintf(aFpOut,"S TAILLEPIX=%d\n",aNbB/8);
      fprintf(aFpOut,"A FORMAT=Totalement Anormal\n");
      fprintf(aFpOut,"*\n");

      ElFclose(aFpOut);
      ElFclose(aFpIn);
      return;
  }

  char aBuf[1000];
  while (1)
  {
     VoidFgets(aBuf,1000,aFpIn);
     fprintf(aFpOut,"%s",aBuf);
     if (aBuf[0] == '*')
     {
        ElFclose(aFpOut);
        ElFclose(aFpIn);
        return;
     }
  }
}


Elise_Tiled_File_Im_2D   ThomParam::file(const char * name_file)
{
    GenIm::type_el aType =  type_im(true,8*TAILLEPIX,false,true);
    return Elise_Tiled_File_Im_2D
           (
                name_file,
                Pt2di(NBCOL,NBLIG),
                aType,
                1,
                Pt2di(NBCOL,NBLIG),
                false,
                true,
                OFFSET,
                false,
                BYTEORD ? (!MSBF_PROCESSOR())  : MSBF_PROCESSOR()
           );
}

/*
 C ENTETE
 A NOM=amiensquadri.16_217.THM
 A FORMAT=Normal
 S TAILLEPIX=1
 S NBCOL=4096
 I COULEUR=8


 C ENTETE
 A OBJECTIF=DIG28c
 A ORIGINE=Camera21
 I MAXIMG=4096
 I MINIMG=0
 I COULEUR=0
 I CAMERA=21
 F EXPOTIME=85
 I FOCALE=28
 F DIAPHRAGME=5.6
 I TDI=17
 A NOM=amiensquadri_b.16_217
 A DATE=19/12/2002 11h:43m:16s:70
 A FORMAT=Normal
 S TAILLEPIX=2
 S NBCOL=4106
 S NBLIG=4096

     BYTEORD=0  // DEF
     BYTEORD=1
*/

ThomParam::ThomParam(const char * name_file)
{

     BYTEORD=0;
     NBLIG=-1;

     ORIGINE = "CameraInconnue";
     OBJECTIF = "OBJECTIFInconnu";
     MAXIMG =-1;
     MINIMG = -1;
     mCAMERA = -1;
     EXPOTIME = -1;
 
     // Nouvellement mis en option pour les cameras tete-queue
     FOCALE = -1;
     DIAPHRAGME = -1;
     TDI=0;
     BIDON=0;

     std::string OPERATION;
     INT NBIMAGES;

     
     OFFSET =
     ThomInitArgs
     (
          name_file,
          LArgMain() << EAM(ORIGINE,"ORIGINE",true)
                     << EAM(OBJECTIF,"OBJECTIF",true)
                     << EAM(MAXIMG,"MAXIMG",true)
                     << EAM(MINIMG,"MINIMG",true)
                     << EAM(mCOULEUR,"COULEUR",true)
                     << EAM(mCAMERA,"CAMERA",true)
                     << EAM(EXPOTIME,"EXPOTIME",true)
                     << EAM(FOCALE,"FOCALE",true)
                     << EAM(DIAPHRAGME,"DIAPHRAGME",true)
                     << EAM(TDI,"TDI",true)
                     << EAM(NOM,"NOM",false)
                     << EAM(DATE,"DATE",true)
                     << EAM(FORMAT,"FORMAT",false)
                     << EAM(TAILLEPIX,"TAILLEPIX",false)
                     << EAM(NBCOL,"NBCOL",false)
                     << EAM(NBLIG,"NBLIG",true)
                     << EAM(MERE,"MERE",true)
                     << EAM(BLANC,"BLANC",true)

                     <<  EAM(OPERATION,"OPERATION",true)
                     <<  EAM(NBIMAGES,"NBIMAGES",true)
                     <<  EAM(BIDON,"BIDON",true)
                     <<  EAM(BYTEORD,"BYTEORD",true)
    );
     OFFSET = 1024 ;
     if (NBLIG==-1)
        NBLIG = NBCOL;

}

Elise_Tiled_File_Im_2D Elise_Tiled_File_Im_2D::Thom
                       (
                            const char * name_file
                       )
{
    ThomParam aTP(name_file);
    return aTP.file(name_file);
}




Im2D_Bits<1> MasqImThom
             (
                const std::string & aNameThom,
                const std::string & aNameFileXML,
                Box2di  &           aBoxFr
             )
{
     ThomParam aTP(aNameThom.c_str());
     Im2D_Bits<1> aIm(aTP.NBCOL,aTP.NBLIG,0);

     cElXMLTree aGlobTree(aNameFileXML);
     Box2di aBox=aGlobTree.Get("usefull-frame")->GetDicaRect();
     ELISE_COPY(rectangle(aBox._p0,aBox._p1),1,aIm.out());


     ELISE_COPY
     (
         rectangle(Pt2di(0,aTP.NBLIG-aTP.TDI-1),Pt2di(aTP.NBCOL,aTP.NBLIG)),
         0,
         aIm.out()
     );

     std::list<cElXMLTree *>  aTreeDef = aGlobTree.GetAll("defect");

     for 
     (
        std::list<cElXMLTree *>::iterator itT=aTreeDef.begin(); 
        itT!=aTreeDef.end() ; 
        itT++
    )
    {
       Box2di aBox=(*itT)->GetDicaRect();
       ELISE_COPY(rectangle(aBox._p0,aBox._p1),0,aIm.out());
    }

    aBoxFr=aGlobTree.Get("dark-frame")->GetDicaRect();
    return aIm;
}


void ThomCorrigeCourrantObscur(Im2D_U_INT2 anIm,const Box2di& aBox)
{
    int aSomIm,aSom1;
    ELISE_COPY
    (
       rectangle(aBox._p0,aBox._p1),
       anIm.in(),
       sigma(aSomIm) | (sigma(aSom1) << 1)
    );

    if (aSom1==0) 
        return;
    int aCourObs = round_ni(aSomIm/REAL(aSom1));
    cout << "COUR OBSCUR = " << aCourObs << "\n";
    ELISE_COPY
    (
      anIm.all_pts(),
      Max(anIm.in()-aCourObs,0),
      anIm.out()
    );
}


int HackToF(int argc,char ** argv)
{
   Pt2di aSz(320,240);
   std::string aName("Test4MPD.bin");
   int aNbByte = 4;
   int aOffset = 0;


   INT aSzFile = sizeofile (aName.c_str());
   INT aSzFrame = (aNbByte * aSz.x * aSz.y);
   int aNbFrame = (aSzFile-aOffset) / aSzFrame;

   ELISE_fp aFP(aName.c_str(),ELISE_fp::READ);
   for (int aKF=0 ; aKF<aNbFrame ; aKF++)
   {
      aFP.seek(aKF*aSzFrame,ELISE_fp::sbegin);
      Im2D_U_INT1 aI1(aSz.x,aSz.y);
      Im2D_U_INT1 aI2(aSz.x,aSz.y);
      Im2D_U_INT1 aI3(aSz.x,aSz.y);
      Im2D_U_INT1 aI4(aSz.x,aSz.y);

      if ((aKF%10==0))
      {
          for (int aY=0 ; aY< aSz.y ; aY++)
          {
              for (int aX=0 ; aX< aSz.x ; aX++)
              {
                  Pt2di aP(aX,aY);
                  aI1.SetI(aP,aFP.fgetc());
                  aI2.SetI(aP,aFP.fgetc());
                  aI3.SetI(aP,aFP.fgetc());
                  aI4.SetI(aP,aFP.fgetc());
              }
          }
      }
      Tiff_Im::CreateFromIm(aI1,"I1-F"+ToString(aKF)+".tif");
      Tiff_Im::CreateFromIm(aI2,"I2-F"+ToString(aKF)+".tif");
      Tiff_Im::CreateFromIm(aI3,"I3-F"+ToString(aKF)+".tif");
      Tiff_Im::CreateFromIm(aI4,"I4-F"+ToString(aKF)+".tif");

      Tiff_Im::CreateFromFonc("I43-F"+ToString(aKF)+".tif",aSz,256*aI4.in()+aI3.in(),GenIm::u_int2);



      // Tiff_Im::CreateFromIm(aI1,"I1-F"+ToString(aKF));

/*
      for (int aK=0 ; aK<8 ; aK++)
      {
          int aC = aFP.fgetc();
          printf("%4d " ,aC);
      }
      printf("\n");
*/
   }

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
Footer-MicMac-eLiSe-25/06/2007*/

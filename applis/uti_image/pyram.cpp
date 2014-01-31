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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>


class PyramVisu
{
	public :

	  PyramVisu(Pt2di sz);

	  Gray_Pal                   Pgray;
          Elise_Set_Of_Palette       SOP;
          Video_Display              Ecr;
          Video_Win                  W;
};

PyramVisu::PyramVisu(Pt2di sz) :
	Pgray	(16),
	SOP	(NewLElPal(Pgray)),
	Ecr	((char *)NULL),
	W	(Ecr,SOP,Pt2di(50,50),sz)
{
	Ecr.load(SOP);
	ELISE_COPY
	(
		W.all_pts(),
		(FX+FY)%255,
		W.ogray()
	);
}


class NivRed
{
	public :
		~NivRed();
		NivRed
		(
			Pt2di   		SZGLOB,
			Pt2di   		DALLES,
			INT 			zoom,
			vector<NivRed *> & 	VB,
			INT			NbBits,
			ElGenFileIm		FINIT,
			bool			VISU ,
			INT				NBCHAN
		);

	

		void Update(INT4 **); 
		void TransfInBd();
		void ViderDalle();
		void Reset();
		void VisuAv();
		void MakeTiff(const string &,Tiff_Im::COMPR_TYPE,Tiff_Im::PH_INTER_TYPE);
		
	protected :
		Pt2di 			SzGlob;
		Pt2di 			Dalles;
		vector<NivRed *> 	fils;

		INT 			zoom;
		NivRed * 		pere;
		INT         		ZoomRel;
		INT         		ZR2;
		INT			TxLoc;
		INT			TyLoc;

		Im2D_INT4      		 LineCur;
		INT **			LC;
		Im2D_U_INT1		BufDalles;
		U_INT1 **		BD;

		INT			Yabs;
		INT			YabsLoc;
		INT			YinPere;
		INT			YinDalle;
		Tiff_Im	*		Tiff;
		PyramVisu *		PyVi;
		INT			NbBits;
		INT			NbBitsPERE;
		bool			visu;
		INT			NbChan;

};


NivRed::~NivRed()
{
	if (Tiff)
	{
		ViderDalle();
		delete Tiff;
	}
	for (INT k= 0; k<(INT) fils.size() ; k++)
		delete fils[k];
}

void 	NivRed::MakeTiff
     	(
		const string & 			Name,
		Tiff_Im::COMPR_TYPE 	MC,
		Tiff_Im::PH_INTER_TYPE  PhInt
	)
{
	if (visu)
	{
		PyVi  = new PyramVisu (Pt2di(TxLoc,TyLoc));
	}
	else
	{
		char buf[20];
		sprintf(buf,"Reduc%d.tif",zoom);
		string NName = Name + buf;
		Tiff = new Tiff_Im
		   	(
				NName.c_str(),
				Pt2di(TxLoc,TyLoc),
				type_u_int_of_nbb(NbBits),
				MC,
				PhInt,
                ElList<Arg_Tiff>() +Arg_Tiff(Tiff_Im::ATiles(Dalles))
			);
	}
}

template <class Type> void Interv(const char * mes,Type * l,INT TxLoc)
{
	INT v0 = 255, v1=0;
     for (INT x=0; x<TxLoc; x++)
     {
	v0 = ElMin(v0, (INT)l[x]);
	v1 = ElMax(v1, (INT)l[x]);
     }
cout << mes << "[" << (INT) v0 << " " << (INT) v1 << "] \n";
}

void NivRed::TransfInBd()
{
	for (INT d=0 ; d<NbChan; d++)
     	for (INT x=0; x<TxLoc; x++)
     	{
         	BD[YinDalle+d*Dalles.y][x] = LC[d][x];
     	}
}                  

void NivRed::ViderDalle()
{
	if (!Tiff)
		return;
	if (YinDalle ==0)
		return;
	INT Dy = YabsLoc - YinDalle;

	Fonc_Num f = BufDalles.in(0);
	for (INT d=1; d<NbChan; d++)
		f = Virgule(f,trans(BufDalles.in(0),Pt2di(0,Dalles.y*d)));

	ELISE_COPY
	(
		rectangle
		(
			Pt2di(0,Dy),
			Pt2di(TxLoc,Dy+Dalles.y)
		),
		trans(f,Pt2di(0,-Dy)),
		(*Tiff).out()
		
	);
}       

void NivRed::VisuAv()
{
	ELISE_COPY
	(
		rectangle
		(
			Pt2di(0,YabsLoc),
			Pt2di(TxLoc,YabsLoc+1)
		),
		(LineCur.in()[Virgule(FX,0)]*255)/((1<<NbBits)-1),
		PyVi->W.out(PyVi->Pgray)
	);
}

void NivRed::Update(INT4 ** l)
{
	Yabs++;
	YinPere++;
	if (l)
	{
		for (INT d=0; d<NbChan ; d++)
		{
			INT4 * LCd = LC[d];
			INT4 * ld = l[d];
			for (INT x=0; x<pere->TxLoc; x++)
			{
		 		LCd[x/ZoomRel] += ld[x];
			}
		}
	}
	if (YinPere==ZoomRel)
	{
	
		for (INT d=0; d<NbChan ; d++)
		{
			INT4 * LCd = LC[d];
			if (NbBits == NbBitsPERE)
			{
		    	for (INT x=0; x<TxLoc; x++)
			 		LCd[x] /= ZR2;
			}
			else if (NbBits < NbBitsPERE)
			{
		    	INT z2 = ZR2 << (NbBitsPERE-NbBits);
		    	for (INT x=0; x<TxLoc; x++)
			 		LCd[x] = (LCd[x] ) /z2;
			}
			else if (NbBits > NbBitsPERE)
			{
		    	INT z2 = ZR2 * ( (1<<NbBitsPERE) -1);
		    	INT mul =  ( (1<<NbBits) -1);
		    	for (INT x=0; x<TxLoc; x++)
			 		LCd[x] = (LCd[x] *mul) /z2;
			}
		}

		if (visu)
		{
			VisuAv();
		}

		for (INT k=0; k<(INT) fils.size(); k++)
			fils[k]->Update(LC);
		TransfInBd();
		YabsLoc++;
		YinDalle++;
		if (YinDalle  == Dalles.y)
                {
                        ViderDalle();
                        YinDalle = 0;
                }          
		for (INT d=0; d<NbChan ; d++)
 			for (INT x=0; x<TxLoc; x++)
                        LC[d][x] =0;
		YinPere=0;
	}
}

NivRed::NivRed
(
	Pt2di   		SZGLOB,
	Pt2di   		DALLES,
	INT 			ZOOM,
	vector<NivRed *> & 	VB,
	INT 			NBBITS,
	ElGenFileIm		FINIT,
	bool			VISU,
	INT				NBCHAN
) :
	SzGlob   	(SZGLOB),
	Dalles	    	(DALLES),
	zoom 	 	(ZOOM),
	pere 	 	(0),
	ZoomRel		(1),
	LineCur  	(1,1),
	BufDalles	(1,1),
	Yabs		(0),
	YabsLoc		(0),
	YinPere		(0),
	YinDalle	(0),
	Tiff		(0),
	NbBits		(NBBITS),
	visu		(VISU),
	NbChan		(NBCHAN)
{
	if (Dalles.x <0) 
		Dalles.x = 1000000000;

	for (INT k=0; k<(INT)	VB.size(); k++)
		if (
			   (zoom % VB[k]->zoom == 0) 
			&& (( VB[k]->zoom != 1)  || (k==0))
		   )
		{
			pere =  VB[k];
			ZoomRel = zoom / pere->zoom;
			ZR2 = ElSquare(ZoomRel);
		}
	if (pere)
	{
		NbBitsPERE = pere->NbBits;	
		pere->fils.push_back(this);
	}
	else
	{
		NbBits = FINIT.NbBits();
	}


	if (! pere)
	{
		TxLoc = SZGLOB.x;
	}
	else
	{
		ELISE_ASSERT(pere!=0,"Inc in NivRed");
		TxLoc = (pere->TxLoc+ZoomRel-1)/ZoomRel;
		BufDalles = Im2D_U_INT1(TxLoc,Dalles.y*NBCHAN,0);
	}
	TyLoc = (SZGLOB.y+zoom-1)/zoom;

	Dalles.x = ElMin(Dalles.x,TxLoc);

	LineCur = Im2D_INT4(TxLoc,NBCHAN,0);
	LC = LineCur.data();
	BD = BufDalles.data();
}

class PyramImag : 	public Simple_OPBuf1<INT,INT>,
			public NivRed
{
	public :
		PyramImag
		(
			ElGenFileIm  			F,
			Tiff_Im::PH_INTER_TYPE  PhInt,
			INT						NBCHAN,
			const string & 			Name,
			vector<INT> 			ZOOMS,
			Pt2di					DALLES,
			INT						NbBits,
			Tiff_Im::COMPR_TYPE		,
			bool					VISU,
			Pt2di					SzMaxVisu
  		);

		void  calc_buf (INT ** output,INT *** input);

		void Finish();

	private :

		static  vector<NivRed *> NoReduc;

		vector<NivRed *>      Reducs;
		Pt2di 				Sz;
};

vector<NivRed *> PyramImag::NoReduc;

PyramImag::PyramImag
(
	ElGenFileIm 		aFileIm,
	Tiff_Im::PH_INTER_TYPE  PhInt,
	INT                     NBCHAN,
	const string &		Name,
	vector<INT> 		ZOOMS,
	Pt2di			DALLES,
	INT			NbBits,
	Tiff_Im::COMPR_TYPE 	MC,
	bool			VISU,
	Pt2di			SzMaxVisu
) :
	NivRed  (aFileIm.Sz2(),DALLES,1,NoReduc,NbBits,aFileIm,false,NBCHAN),
	Sz		(aFileIm.Sz()[0],aFileIm.Sz()[1])
{
	STDSORT(ZOOMS.begin(),ZOOMS.end());

	if (VISU)
	{
		INT z0 = ZOOMS.back();
		INT zv=z0;
		while ((zv*SzMaxVisu.x<Sz.x) || (zv*SzMaxVisu.y<Sz.y))
			zv += z0;
		ZOOMS.push_back(zv);
	}

 	Reducs.push_back(this);
	for (INT k=0; k<(INT) ZOOMS.size() ; k++)
	{
		NivRed * nr = new NivRed 
				  (
					SzGlob,
					Dalles,
					ZOOMS[k],
					Reducs,
					NbBits,
					aFileIm,	
					VISU && (k==((INT)ZOOMS.size()-1)),
					NBCHAN
				  );
		Reducs.push_back(nr);
		nr->MakeTiff(Name,MC,PhInt);
	}
	
}

void  PyramImag::calc_buf (INT ** output,INT *** input)
{
	ELISE_ASSERT
	(
		    (x0()==0)
		&&  (y0()==0)
		&&  (x1()==TxLoc)
		&&  (y1()==TyLoc),
		"Inc in PyramImag::calc_buf"
	);

	INT * data[10];
 	for (INT d=0 ; d< NbChan ; d++)
		data[d] = input[d][0];


	for (INT k=0; k<(INT)fils.size(); k++)
		fils[k]->Update(data);
	for (INT x=0 ; x<TxLoc ; x++)
		output[0][x] = 0;
}


std::string Substitute
            (
                 const std::string & aStr,
                 const std::string & aOld,
                 const std::string & aNew
             )
{
   std::string aRes = aStr;
   std::string::size_type aPosition = aRes.rfind(aOld);
   ELISE_ASSERT(aPosition != std::string::npos,"Dont find SubString in Substitute");
   aRes.replace(aPosition,aNew.length(),aNew);
   return aRes;
}



int main(int argc,char ** argv)
{
/*
	for (INT k=0; k<argc; k++)
		cout << "{" << argv[k] << "} \n";
	cout << "-----------\n";
*/


	string SPrefix = "";
	string Name;
    	vector<INT> Zooms;
	Pt2di  Dalles(256,256);
	INT NbBits = -1;
	string Compr("Id"); // Par defaut on reprend le meme mode de compr.
	Pt2di	SzMaxVisu(700,500);
	INT  PckbScroll = 0;
	INT  Visu = 0;
        INT  Invert = 0;

        INT WhitePur = 255;
        INT BlackPur = 0;
        INT Sep2RGB = 0;

	std::string r("r");
	std::string v("v");
	std::string b("b");

	ElInitArgMain
	(
		argc,argv,
		LArgMain() 	<< EAM(Name) 
				<< EAM(Zooms),
		LArgMain() 	<< EAM(Dalles,"Dalles",true)
                                << EAM(SPrefix,"NameOut",true)
				<< EAM(NbBits,"NbBits",true)
				<< EAM(Compr,"Compr",true)
				<< EAM(SzMaxVisu,"SzMaxVisu",true)
				<< EAM(Visu,"Visu",true)
				<< EAM(PckbScroll,"PckbScroll",true)
                                << EAM(Invert,"Invert",true)
                                << EAM(WhitePur,"WhitePur",true)
                                << EAM(BlackPur,"BlackPur",true)
                                << EAM(Sep2RGB,"Sep2RGB",true)
                                << EAM(r,"r",true)
                                << EAM(v,"v",true)
                                << EAM(b,"b",true)
			
	);	

        if (SPrefix=="")
           SPrefix = StdPrefix(Name);


	Tiff_Im TIFF = Tiff_Im::StdConv(Name); 
        INT NbBitsOri = TIFF.NbBits();
	if (NbBits==-1)
	   NbBits = NbBitsOri;


	Tiff_Im::COMPR_TYPE  MC = TIFF.mode_compr();
	if (Compr != "Id")
		MC = Tiff_Im::mode_compr(Compr);


        Tiff_Im::PH_INTER_TYPE aPhotoInt = TIFF.phot_interp();

	if (PckbScroll)
	{
           NbBits = 8;
           MC = Tiff_Im::PackBits_Compr;
           Dalles.x = 512;
           Visu = 1;
	}



         Fonc_Num FInput = TIFF.in();

         if (Sep2RGB)
         {
	      std::string Namev = Substitute(Name,r,v);
	      std::string Nameb = Substitute(Name,r,b);
              
	      cout << "Namev = " << Namev.c_str() << "\n";
	      cout << "Nameb = " << Nameb.c_str() << "\n";

              Tiff_Im  Tiffv = Tiff_Im::StdConv(Namev.c_str());
              Tiff_Im  Tiffb = Tiff_Im::StdConv(Nameb.c_str());

              FInput = Virgule(TIFF.in() ,Tiffv.in(),Tiffb.in());
              aPhotoInt = Tiff_Im::RGB;
         }

         INT Vmax = (1<<NbBitsOri) -1;


        if ((WhitePur != 255) || (BlackPur != 0))
        {
            REAL Bl = BlackPur * Vmax / 255.0;
            REAL Wh = WhitePur * Vmax / 255.0;
            FInput =  Max(0,Min(Vmax,(FInput-Bl) * (Vmax/(Wh-Bl))));
        }


        if (Invert)
           FInput = Vmax -FInput;

	Fonc_Num  f = 	create_op_buf_simple_tpl
			(
				new PyramImag
				    (
					TIFF,
					aPhotoInt,
					FInput.dimf_out(),
					SPrefix,
					Zooms,
					Dalles,
					NbBits,
					MC,
					(Visu!=0),
					SzMaxVisu
				    ),
				0,
				FInput,
				1, // L'output est bidon, donc peu importe
				Box2di(Pt2di(0,0),Pt2di(0,0))
			);               


	
	ELISE_COPY
	(
		TIFF.all_pts(),
		f,
		Output::onul()
	);

    return 1;
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

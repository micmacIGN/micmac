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

#if (ELISE_VW_W95NT_API)
/*
	PB : 

	[1] Quand on creer une fenetre TX,TY, windows interprete
	cela comme la taille Y COMPRIS toutes les decorations 
	eventuelles. Comment connaitre la taille de ces differentes
	"enjolivures" pour augmenter la taille demandee afin 
	que TX,TY soit effectivement la taille accessible au client.
	
	On doit pouvoir le coder en dur en faisant qq essais mais :
		a- c'est degueulasse
		b- a tout les cas ca depend de la version de windows

	[2] Apparemment, dans SetDIBitsToDevice, le coin "bas-droit"
	est a donner en delta / au coin haut droit de la Fenetre.
	Ca semble la convention la plus bete du monde, mais
	ca ne marche que comme ca. Ai-je bien compris. 

	Y a-t-il une doc qui clarifie cela.

	[3] Quand on utilise une DIB sur 16 bits il apparait, sur
	ma machine que windows l'interprete en 5-5-5. Lorsque l'on
	declare une DIB sur 24 bits, il apparait que windows l'interprete
	en RGB. Tous cela est a peu pres OK, mais :

		- est-ce toujours comme cela (par exemple X11 peut suivant
		les serveur et les configs tourner en 5-5-5 ou en 6-5-5;
		le 24 bits peut etre du BGR suivant que l'on soit sur
		une machine MSBF ou LSBF ..., il peut y avoir un byte
		de padding ...)

		- si cela doit varier, y a-t-il un protocole permettant
		de connaitre la codification RVB du bitmap;


		- par ailleurs, est-on absolument certain, comme cela
		semble logique, que a partir du moment ou on prend un
		DIB 16 ou 24 bits, on fonctionne en vraies couleurs
		(du moins sur ecran 16 ou 24 bits, il faudra que je regarde
		ce qui se passe sur ecran 8 bits avec des bitmaps 16/24,
		bof ... => si l'ecran est en 8bits, mieux vaut avoir une
		DIB 8 bits et jouer sur les palettes comme d'hab).

  [3] Comment savoir si windows est lance en 8, 16 ou 24 bits
  (sachant que le 4 on lui pisse dessus). Question rendue
  necessaire par le fait que "a priori" :

		- il est dommage d'avoir des DIB avec moins de bits que
		l'ecran (car perte d'info des que l'on passe par eux)

		- il est dommage de gerer des DIB avec plus de bits
		(gachis de memoire et, a voir, probable perte de temps
		car le "serveur" devra tout reconvertir).
  
	[4] Comment creer un device associe a une DIB.



*/


#include <cwindows>  
#include <cwinuser>
#include <cstdio>
#include <cstdlib>
#include <string>

#define IDM_EXIT           100
#define IDM_TEST           200
#define IDM_ABOUT          301

#define IDC_TEXT           101
#define IDC_EDIT           102
#define IDC_BTN            103

LRESULT CALLBACK WndProc  (HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK About    (HWND, UINT, WPARAM, LPARAM);


#if defined (WIN32)
	#define IS_WIN32 TRUE
#else
	#define IS_WIN32 FALSE
#endif

#define IS_NT      IS_WIN32 && (BOOL)(GetVersion() < 0x80000000)
#define IS_WIN32S  IS_WIN32 && (BOOL)(!(IS_NT) && (LOBYTE(LOWORD(GetVersion()))<4))
#define IS_WIN95   (BOOL)(!(IS_NT) && !(IS_WIN32S)) && IS_WIN32

/**************************************/
extern HINSTANCE hInst;

class DVD  // Data_Video_Display
{
	friend class DVV;


	WNDCLASS	_wc;
	HINSTANCE	_hInstance; 
	HINSTANCE	_hPrevInstance;
    LPTSTR		_lpCmdLine;
	int			_nCmdShow;

	static const char * _AppName;
	static const char * _Elise_Name;
	BOOL RegisterWin95() const;
	bool _init_carac;
	int  _bits_per_pixel;
	int  _nb_oct_pp;

	void init_carac(HDC);


public :
	DVD
	(
		HINSTANCE hInstance, 
		HINSTANCE hPrevInstance,
        LPTSTR lpCmdLine, 
		int nCmdShow	
	);
	static int reaff(); 
};

class DVV   // Data_Video_Window 
{

	DWORD		_dwstyle;
	HWND		_hWnd;	 // La Fenetre proprement dite
	HDC   _hdc;			// Le Device Cont associe a _hWnd
// image de 1 ligne
	BITMAPINFO * _bmi;	// L'info sur la DIB + les data (= _im)
	HBITMAP _hbm;		// un handle sur _bmi
	void flush_buf_b1(); // vide le buffer de 1 ligne dans le MemDC
	void flush_buf_b1(HDC); 
// garde memoire de l'ecriture courante dans le tampon de 1 ligne
	int _last_x;
	int _last_y;
	int _last_x0;

//  Bitmap + Device contexte memorisant la fenetre
     BITMAPINFOHEADER  _bi;
     BITMAPINFOHEADER* _lpbi;
     HBITMAP           _hBitmap;
     HDC			   _hMemDC;
     HANDLE            _hDIB;


	unsigned char * _im; // bits du DIB 

	int   _txF;    // taille utilisateur des fenetre
	int   _tyF;
	int   _txB;    // taille des bitmaps, eventuellement diff pour
	int   _tyB;    // pour probleme d'alignement

	int   _nb_oct; // nombre d'octet par pixel (1,2 ou 3)

	
	typedef enum
	{
		NB_COLOUR = 256  // Avant de gerer les palettes
	};

public :

	~DVV();
	DVV
	(
		DVD* dvd,
		int tx,
		int ty
	);
	static inline int unsigned short to_16(int r,int g, int b)
	{
		return
			(r/8) << 10
		|   (g/8) << 5
		|   (b/8)         ;
	}


// Raffraichit la fenetre avec le rectangle du bitmap
	void show_rect(int x1,int y1,int x2,int y2);
	void show_rect_glob();

// juste pour test/mise au point; sans doute tres lent
	void bitm_set_rgb(int x,int y,int r,int g,int b);


	
// DEBUG
	void show_int(int v,int x,int y);
	// Ouverture tres provisoire, a moyen terme il est hors
	// de question de laisser voir au reste du monde la
	// misere de Windows

	HDC		hdc()			{ return _hdc;}
	HDC     hmdc()          { return _hMemDC;}

//	HWND	hWnd()		{ return _hWnd;}
//	HBITMAP   hbm()		{ return _hbm;}
//	BITMAPINFO * bmi()	{ return _bmi;}
	static DVV * Elise_Object(HWND h)
	{
		return (DVV *) GetProp(h,"EliseThis");
	};
};


HINSTANCE hInst;   // current instance

const char * DVD::_AppName     = "MpdApp";
const char * DVD::_Elise_Name = "Fenetre Elise";


/******************************************************/
/******************************************************/
/***                                                 **/
/***    Data_Video_Display                           **/
/***                                                 **/
/******************************************************/
/******************************************************/

//    Ou du moins de ce qui en tient lieu, apres  
//     tout c'est WINDOWS.                 
//                                                 

DVD::DVD
(
 		HINSTANCE hInstance, 
		HINSTANCE hPrevInstance,
        LPTSTR lpCmdLine, 
		int nCmdShow	
)
{
	_hInstance		= hInstance;
	_hPrevInstance	= hPrevInstance;
	_lpCmdLine		= lpCmdLine;
	_nCmdShow		= nCmdShow;

	_wc.style         = CS_SAVEBITS 
						// CS_HREDRAW | CS_VREDRAW								
					;
	_wc.lpfnWndProc   = (WNDPROC)WndProc;			
	_wc.cbClsExtra    = 0;                      
	_wc.cbWndExtra    = 0;                      
	_wc.hInstance     = hInstance;              
	_wc.hIcon         = LoadIcon( hInstance, _AppName ); 
	_wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
	_wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
	_wc.lpszMenuName  = _AppName;              
	_wc.lpszClassName = _AppName;              

   if ( IS_WIN95 )
   {
      if ( !RegisterWin95() )
         e-xit(-1);
   }
   else if ( !RegisterClass(&_wc ) )
      e-xit(-1);

   hInst = hInstance; 
   _init_carac = false;
}

void DVD::init_carac(HDC hdc)
{
	if (! _init_carac)
	{
		_init_carac = true;
		_bits_per_pixel = GetDeviceCaps(hdc,BITSPIXEL);
		if (
				(_bits_per_pixel %8)
			||	(_bits_per_pixel == 0)
			||	(_bits_per_pixel > 32)
			)
			e-xit(-1);
		_nb_oct_pp = _bits_per_pixel /8;

	}
}

BOOL DVD::RegisterWin95() const
{

   WNDCLASSEX wcex;

   wcex.style         = _wc.style;
   wcex.lpfnWndProc   = _wc.lpfnWndProc;
   wcex.cbClsExtra    = _wc.cbClsExtra;
   wcex.cbWndExtra    = _wc.cbWndExtra;
   wcex.hInstance     = _wc.hInstance;
   wcex.hIcon         = _wc.hIcon;
   wcex.hCursor       = _wc.hCursor;
   wcex.hbrBackground = _wc.hbrBackground;
   wcex.lpszMenuName  = _wc.lpszMenuName;
   wcex.lpszClassName = _wc.lpszClassName;

   // Added elements for Windows 95.
   //...............................
   wcex.cbSize = sizeof(WNDCLASSEX);

#if (Compiler_Visual_6_0)
   wcex.hIconSm = (HICON__ *) LoadImage(wcex.hInstance, _wc.lpszClassName, 
                            IMAGE_ICON, 16, 16,
                            LR_DEFAULTCOLOR );
#else
   wcex.hIconSm =  LoadImage(wcex.hInstance, _wc.lpszClassName, 
                            IMAGE_ICON, 16, 16,
                            LR_DEFAULTCOLOR );
#endif
			
   return RegisterClassEx( &wcex );

}

/******************************************************/
/******************************************************/
/***                                                 **/
/***    Data_Video_Window                            **/
/***                                                 **/
/***                                                 **/
/******************************************************/
/******************************************************/


DVV::DVV
(
		DVD*	dvd,
		int		tx,
		int		ty
)
{
	_dwstyle = WS_OVERLAPPEDWINDOW;



	_txF = _txB = tx;
	_tyF = _tyB = ty;

			// Vu chez Frankies, Probabl Pb d'alignement .
	while (_txB%4) _txB ++;
	while (_tyB%4) _tyB ++;

//===============================
//   CREATION DE LA FENETRE
//================================

	{
		RECT r;
		r.left = 0;
		r.top = 0;
		r.right = _txF;
		r.bottom = _tyF;

			// renvoie les coordonnees de la Fenetre, y compris
			// enjolivures dans le systeme de coordonnees utilisateur
		AdjustWindowRect(&r,_dwstyle,FALSE);

		_hWnd = CreateWindow
		(
			DVD::_AppName, 
			DVD::_Elise_Name,    
            _dwstyle, 
            CW_USEDEFAULT, 
			0, 
            r.right-r.left,
			r.bottom-r.top,  
            NULL,              
            NULL,              
            dvd->_hInstance,          
            NULL               
         );
	
	}

   if ( !_hWnd ) 
      e-xit(-1);

   ShowWindow( _hWnd, dvd->_nCmdShow ); 
   UpdateWindow( _hWnd );   
   _hdc = GetDC(_hWnd);
   dvd->init_carac(_hdc);
   _nb_oct = dvd->_nb_oct_pp;

//===============================
//   CREATION DU BITMAP DE 1 LIGNE
//================================

   _last_y = -1;
   _bmi =	(BITMAPINFO *) 
			malloc 
			(
				  sizeof (BITMAPINFO)
				+ NB_COLOUR*sizeof(RGBQUAD)
				+ _txB*1*_nb_oct 
			);

	_bmi->bmiHeader.biSize = (DWORD) sizeof (BITMAPINFOHEADER) ;
	_bmi->bmiHeader.biWidth = (LONG) _txB ;
	_bmi->bmiHeader.biHeight = (LONG) 1 ; // -ny
	_bmi->bmiHeader.biPlanes = (WORD) 1 ;
	_bmi->bmiHeader.biBitCount = (WORD) (8 * _nb_oct);
	_bmi->bmiHeader.biCompression = (DWORD) BI_RGB ;
	_bmi->bmiHeader.biSizeImage = (DWORD) 0 ;
	_bmi->bmiHeader.biXPelsPerMeter = (LONG) 0 ;
	_bmi->bmiHeader.biYPelsPerMeter = (LONG) 0 ;
	_bmi->bmiHeader.biClrUsed = (LONG) NB_COLOUR ;
	_bmi->bmiHeader.biClrImportant = (LONG) NB_COLOUR ;
	
	// sans doute inutile avant de gerer les palettes, 
	// surement a modifier avec gestion des palettes,
	// surement inutile en mode 16 ou 24 bits
	/*
		{
			int x;
			for (x=0; x<128; ++x) 
			{
				_bmi->bmiColors[x] . rgbBlue = 2*x ;
				_bmi->bmiColors[x] . rgbGreen = 2*x ;
				_bmi->bmiColors[x] . rgbRed = 2*x ;
				_bmi->bmiColors[x] . rgbReserved = 0 ;
			}
		}
	*/

	int offset_im = _bmi->bmiHeader.biSize 
					+ NB_COLOUR*sizeof(RGBQUAD);

	_im =		(unsigned char*)_bmi + offset_im;
	_hbm = CreateDIBitmap
			(
				_hdc,
				(LPBITMAPINFOHEADER)_bmi,
				(DWORD)CBM_INIT,
				(LPSTR)_bmi + offset_im,
				(LPBITMAPINFO)_bmi,
				DIB_RGB_COLORS 
			);

	if (_hbm==0)
		e-xit(-1);


//================================================
//   CREATION DU DC/BITMAP          
// Copie betement dans le "Richard Simon"
// Absolument pas bite une seule ligne de ce
// charabia. Apparament ca marche.
//================================================

     _bi.biSize     = sizeof(BITMAPINFOHEADER);
     _bi.biWidth    = _txB;
     _bi.biHeight   = _tyB;
     _bi.biPlanes   = 1;
     _bi.biBitCount = 8*_nb_oct;
     _bi.biCompression   = BI_RGB;
     _bi.biSizeImage     = 0;
     _bi.biXPelsPerMeter = 0;
     _bi.biYPelsPerMeter = 0;
     _bi.biClrUsed       = 0;
     _bi.biClrImportant  = 0;

     // Create DIB.
     //............
     _hBitmap = CreateDIBitmap( _hdc, &_bi, 0L, NULL,NULL, 0 );

     // Allocate memory for BITMAPINFO structure.
     //..........................................
     _hDIB    = GlobalAlloc( GHND, 
                            sizeof( BITMAPINFOHEADER )+
                            NB_COLOUR * sizeof( RGBQUAD ) );

     _lpbi = (BITMAPINFOHEADER*)GlobalLock( _hDIB );

     // Copy bi to top of BITMAPINFO structure.
     //........................................
     *_lpbi = _bi;

     // Use GetDIBits() to init bi struct data.
     //........................................
     GetDIBits( _hdc, _hBitmap, 0, 50, NULL, 
                (LPBITMAPINFO)_lpbi, DIB_RGB_COLORS );
     GlobalUnlock( _hDIB );

     // Create a memory device context 
     // and select the DIB into it.
     //...............................
     _hMemDC = CreateCompatibleDC( _hdc );
     SelectObject( _hMemDC, _hBitmap );

	 SetProp(_hWnd,"EliseThis",this);
}


DVV::~DVV()
{
	DeleteObject (_hbm) ;
	ReleaseDC(_hWnd,_hdc);
	DeleteDC( _hMemDC );
    GlobalFree( _hDIB );
}


void DVV::flush_buf_b1(HDC hdc)
{
	SetDIBitsToDevice
	(
		hdc, 
		_last_x0,_last_y, // x,y dans cible
		(DWORD)(_last_x-_last_x0),(DWORD) (1), // larg-haut 
		_last_x0,0,          // dans src
		0, (UINT)_tyB, 
		_im, _bmi, 
		(UINT) DIB_RGB_COLORS
	);
}


void DVV::flush_buf_b1()
{
	if ((_last_y>=0) && (_last_x > _last_x0))
	{
		flush_buf_b1(_hdc);
		flush_buf_b1(_hMemDC);
		_last_x0 = _last_x;
	}

}


void DVV::bitm_set_rgb(int x,int y,int r,int g,int b)
{
	if (
			(y!= _last_y)
		 || (x != _last_x)
		)
	{
		flush_buf_b1();
		_last_x0 = x;
	}
	_last_x =x+1;
	_last_y = y;

	switch(_nb_oct)
	{
		case 1:
			_im[x] = (r+g+b) /6;
		break;

		case 2:
			((unsigned short *) _im)[x] = to_16(r,g,b);
		break;

		case 3 :
		{
			unsigned char *i = _im + 3*x; 
			i[0] = r;
			i[1] = g;
			i[2] = b;
		}
		break;

		case 4 :
		{
			unsigned char *i = _im + 4*(x+_txB*y); 
			i[0] = r;
			i[1] = g;
			i[2] = b;
		}
		break;
		default :
			e-xit(-1);
	}

}

void DVV::show_rect(int x1,int y1,int x2,int y2)
{
    if(!	BitBlt
			( 
				_hdc,  // cible
				x1, y1, // x,y cible
				x2-x1, y2-y1,  // largeur hauteur
			  _hMemDC, // src 
				x1, y1,  // x y src
				SRCCOPY 
			)
	)
		e-xit(-1);



}
void DVV::show_rect_glob()
{
	show_rect(0,0,_txF,_tyF);
}


void DVV::show_int(int v,int x,int y)
{
	char buf[1000];
	sprintf(buf,"%d",v);
	SetTextColor(_hdc,RGB(255,0,0));
    TextOut(_hdc,x,y,buf,strlen(buf));
}


int DVD::reaff()
{
	MSG      msg;
	while( GetMessage( &msg, NULL, 0, 0) )   
	{
      TranslateMessage( &msg ); 
      DispatchMessage( &msg );  
	}

	return( msg.wParam );
}


LRESULT CALLBACK WndProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{

   switch ( uMsg ) 
   {
   
	   
		case WM_PAINT :
		{
			DVV * win = DVV::Elise_Object(hWnd);
			if (win)
			{
				win->show_rect_glob();
			}
		}
		break;

		case WM_DESTROY :
              PostQuitMessage(0);
        break;

		default :
            return( DefWindowProc( hWnd, uMsg, wParam, lParam ) );
   }

   return( 0L );
}
#endif





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

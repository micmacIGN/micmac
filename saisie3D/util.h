#ifndef UTIL_H
#define UTIL_H

#ifndef PI
#define PI  3.14159265358979323846
#endif

template <class Type> inline Type ElMax (Type v1,Type v2) {return (v1>v2) ? v1 : v2;}
template <class Type> inline Type ElMin (Type v1,Type v2) {return (v1<v2) ? v1 : v2;}

//! Color components type (R,G and B)
typedef unsigned char colorType;

//! Max value of a color component
const unsigned char MAX_COLOR_COMP = 255;

namespace mmColor
{
    // Predefined colors
    static const colorType white[3]			=	{MAX_COLOR_COMP,MAX_COLOR_COMP,MAX_COLOR_COMP};
    static const colorType lightGrey[3]		=	{200,200,200};
    static const colorType darkGrey[3]		=	{MAX_COLOR_COMP/2,MAX_COLOR_COMP/2,MAX_COLOR_COMP/2};
    static const colorType red[3]			=	{MAX_COLOR_COMP,0,0};
    static const colorType green[3]			=	{0,MAX_COLOR_COMP,0};
    static const colorType blue[3]			=	{0,0,MAX_COLOR_COMP};
    static const colorType darkBlue[3]		=	{0,0,MAX_COLOR_COMP/2};
    static const colorType magenta[3]		=	{MAX_COLOR_COMP,0,MAX_COLOR_COMP};
    static const colorType cyan[3]		    =	{0,MAX_COLOR_COMP,MAX_COLOR_COMP};
    static const colorType orange[3]		=	{MAX_COLOR_COMP,MAX_COLOR_COMP/2,0};
    static const colorType black[3]			=	{0,0,0};
    static const colorType yellow[3]		=	{MAX_COLOR_COMP,MAX_COLOR_COMP,0};

    // Predefined materials
    static const float bright[4]			    =	{1.0f,1.0f,1.0f,1.0f};
    static const float lighter[4]			    =	{0.83f,0.83f,0.83f,1.0f};
    static const float light[4]				    =	{0.66f,0.66f,0.66f,1.0f};
    static const float middle[4]			    =	{0.5f,0.5f,0.5f,1.0f};
    static const float dark[4]				    =	{0.34f,0.34f,0.34f,1.0f};
    static const float darker[4]			    =	{0.17f,0.17f,0.17f,1.0f};
    static const float night[4]				    =	{0.0F,0.0F,0.0F,1.0F};

    // Default foreground color
    static const colorType defaultColor[3]			=   {MAX_COLOR_COMP,MAX_COLOR_COMP,MAX_COLOR_COMP};
    static const colorType defaultBkgColor[3]		=   {10,102,151};
    static const colorType defaultHistBkgColor[3]	=   {51,0,51};
    static const colorType defaultLabelColor[3]		=   {255,255,0};
}

//! View orientation
enum MM_VIEW_ORIENTATION {  MM_TOP_VIEW,	/**< Top view (eye: +Z) **/
                            MM_BOTTOM_VIEW,	/**< Bottom view **/
                            MM_FRONT_VIEW,	/**< Front view **/
                            MM_BACK_VIEW,	/**< Back view **/
                            MM_LEFT_VIEW,	/**< Left view **/
                            MM_RIGHT_VIEW,	/**< Right view **/
                            MM_ISO_VIEW_1,	/**< Isometric view 1: front, right and top **/
                            MM_ISO_VIEW_2	/**< Isometric view 2: back, left and top **/
};

#endif // UTIL_H

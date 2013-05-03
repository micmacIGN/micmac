#ifndef GUI_PARAMETERS_H
#define GUI_PARAMETERS_H

//! This class manages the persistent parameters
/** Values of persistent parameters are stored by the system
    from one execution of CloudCompare to the next.
**/
class mmGui
{
public:

    //! GUI parameters
    struct ParamStruct
    {
        //! Light diffuse color (RGBA)
        float lightDiffuseColor[4];
        //! Light ambient color (RGBA)
        float lightAmbientColor[4];
        //! Light specular color (RGBA)
        float lightSpecularColor[4];

        //! Default mesh diffuse color (front)
        float meshFrontDiff[4];
        //! Default mesh diffuse color (back)
        float meshBackDiff[4];
        //! Default mesh specular color
        float meshSpecular[4];

        //! Default text color
        unsigned char textDefaultCol[3];
        //! Default 3D points color
        unsigned char pointsDefaultCol[3];
        //! Background color
        unsigned char backgroundCol[3];
        //! Histogram background color
        unsigned char histBackgroundCol[3];
        //! Labels color
        unsigned char labelCol[3];
        //! Bounding-boxes color
        unsigned char bbDefaultCol[3];
        //! Use background gradient
        bool drawBackgroundGradient;
        //! Decimate meshes when moved
        bool decimateMeshOnMove;
        //! Decimate clouds when moved
        bool decimateCloudOnMove;

        //! Color scale option: always show '0'
        bool colorScaleAlwaysShowZero;
        //! Color scale option: always symmetrical
        /** This only applies to signed scalar fields.
        **/
        bool colorScaleAlwaysSymmetrical;
        //! Color scale square size
        unsigned colorScaleSquareSize;

        //! Default displayed font size
        unsigned defaultFontSize;
        //! Displayed numbers precision
        unsigned displayedNumPrecision;
        //! Labels transparency
        unsigned labelsTransparency;

        //! Default constructor
        ParamStruct();

        //! Copy operator
        ParamStruct& operator =(const ParamStruct& params);

        //! Resets parameters to default values
        void reset();

        //! Loads from persistent DB
        void fromPersistentSettings();

        //! Saves to persistent DB
        void toPersistentSettings();
    };

    //! Returns the stored values of each parameter.
    static const ParamStruct& Parameters();

    //! Sets GUI parameters
    static void Set(const ParamStruct& params);

    //! Release unique instance (if any)
    static void ReleaseInstance();

protected:

    //! Parameters set
    ParamStruct params;

};

//! Display context
struct glDrawContext
{
    unsigned short flags;       //drawing options (see below)
    int glW;                    //GL screen width
    int glH;                    //GL screen height
    //ccGenericGLDisplay* _win;   //GL window ref.

    //default materials
    //ccMaterial defaultMat; //default material
    float defaultMeshFrontDiff[4];
    float defaultMeshBackDiff[4];
    unsigned char pointsDefaultCol[3];
    unsigned char textDefaultCol[3];
    unsigned char labelDefaultCol[3];
    unsigned char bbDefaultCol[3];

    //decimation option
    bool decimateCloudOnMove;

    //information on displayed color scale
    //ccScalarField* sfColorScaleToDisplay;
    bool greyForNanScalarValues;
    char colorRampTitle[256];

    //for displaying text
    unsigned dispNumberPrecision;

    //for displaying labels
    unsigned labelsTransparency;

    //VBO
    //vboStruct vbo;

    //transparency
    //GLenum sourceBlend;
    //GLenum destBlend;

    //Default constructor
    glDrawContext()
    : flags(0)
    , glW(0)
    , glH(0)
    //, _win(0)
    , decimateCloudOnMove(true)
    //, sfColorScaleToDisplay(0)
    , greyForNanScalarValues(true)
    , dispNumberPrecision(6)
    , labelsTransparency(100)
    //, sourceBlend(GL_SRC_ALPHA)
    //, destBlend(GL_ONE_MINUS_SRC_ALPHA)
    {
        colorRampTitle[0]=0;
    }
};

#endif // GUI_PARAMETERS_H

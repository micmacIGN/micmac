#ifndef __3DOBJECT__
#define __3DOBJECT__

#include "Settings.h"
#include "mmglu.h"
/* SELECTION_MODE is defined in cMasq3D_enums.h */
#include "../../geom3d/cMasq3D_enums.h"

#define QMaskedImage cMaskedImage<QImage>

#include "general/errors.h"

#ifdef USE_MIPMAP_HANDLER
	#include "MipmapHandler.h"
#endif

typedef enum // Attention repercutions sur QT ... TODO � regler
{
  qEPI_NonSaisi,	// 0
  qEPI_Refute,		// 1
  qEPI_Douteux,		// 2
  qEPI_Valide,		// 3
  qEPI_NonValue,	// 4
  qEPI_Disparu,		// 5
  qEPI_Highlight	// 6
} qEtatPointeImage;



//! Interaction mode (only in 3D)
enum INTERACTION_MODE {
    TRANSFORM_CAMERA,
    SELECTION
};

enum LINE_STYLE
{
    LINE_NOSTIPPLE,
    LINE_STIPPLE
};


// TODO GERER les etats avec des flags
enum object_state {
    state_default,
    state_overed,
    state_selected,
    state_highlighted,
    state_invible,
    state_disabled,
    state_COUNT
};

enum point_geometry {
    Geom_simple_circle,
    Geom_double_circle,
    Geom_epipolar,
    Geom_cross,
    no_geometry
};

#define ErrPoint cPoint(QPointF(-400000.,-400000.));

class cObject
{
    public:
        cObject();
        cObject(QVector3D pt, QColor color_default);
        virtual ~cObject();

        const QString   & name()        const { return _name;     }
        const QVector3D & getPosition() const { return _position; }
        const QVector3D & getRotation() const { return _rotation; }
        const QVector3D & getScale() const { return _scale;    }
        const QColor    & getColor() const;
        bool    isVisible();
        bool    isSelected()    { return (state() == state_selected);}

        void setName(QString name)          { _name = name;     }
        void setPosition(QVector3D const &aPt);
        void setRotation(QVector3D const &aPt)  { _rotation = aPt;  }
        void setColor(QColor const &aCol, object_state state = state_default)   { _color[state] = aCol;    }
        void setScale(QVector3D aScale)         { _scale = aScale; }
        void setVisible(bool aVis)          { setState(aVis ? state() == state_invible ? state_default : state() : state_invible); }
        void setSelected(bool aSel)         { setState(aSel ? state_selected : state_default);}

        cObject & operator = (const cObject &);

        object_state   state() const;
        void     setState(object_state state);

        cObject* child(int id = 0);

        int		 nbChild(){return _children.size();}

        void	 addChild(cObject* child);

        void	 removeChild(cObject* child);

        void	 replaceChild(int id,cObject* child);

        virtual	 cObject* parent() const;

        virtual	 void	 setParent(cObject* parent);

protected:

        QString _name;

        QVector3D  _position;

        QVector3D   _rotation;

        QColor  _color[state_COUNT];
        QVector3D   _scale;

        float		_alpha;

        object_state   _state;

        QVector< cObject* > _children;

        cObject*			_parent;
};

class cObjectGL : public cObject
{
    public:

        cObjectGL();

        cObjectGL(QVector3D pos, QColor color_default) :
            cObject(pos, color_default),
            _lineWidth(1),
            _glError(0)
        {}

        virtual ~cObjectGL(){}

        virtual void draw()=0;

        void    setLineWidth(float width) { _lineWidth = width; }

        GLenum glError() const
        {
            return _glError;
        }

        void setGlError(const GLenum &glError)
        {
            _glError = glError;
        }


    protected:

        float   _lineWidth;

        void    setGLColor();

        void    enableOptionLine();

        void    disableOptionLine();

        float   getHalfViewPort();

private:

        GLenum _glError;
};

class cPoint : public cObjectGL, public QPointF
{
    public:
    cPoint(QPointF pos = QPointF(0.f,0.f),
           QString name = "",
           bool showName   = false,
           int  statePoint = qEPI_NonValue,
           int  pointGeometry = Geom_simple_circle,
           bool isSelected = false,
           QColor color = Qt::red,
           QColor selectionColor = Qt::blue,
           float diameter = 5.f,
           bool  highlight  = false,
           bool  drawCenter = true);

        void draw();

        void setPointState(int state){ _pointState = state; }
        void setPointGeometry(int g) { _pointGeometry = g;  }
        float diameter()             { return _diameter;    }
        void setDiameter(float val)  { _diameter = val;     }
        int  pointState() const      { return _pointState;  }
        int  pointGeometry() const   { return _pointGeometry;  }
        void showName(bool show)     { _bShowName = show;   }

        bool highlight() const       { return _highlight;   }
        bool showName() const        { return _bShowName;   }
        void setHighlight(bool hl)   { _highlight = hl;     }
        void switchHighlight()       { _highlight = !_highlight; }
        void drawCenter(bool aBool)  { _drawCenter = aBool; }

        void setPosition(QPointF pos);

        void setEpipolar(QPointF pt1, QPointF pt2);

        void glDraw();

        QColor colorPointState();

        virtual	 void	 setParent(cObject* parent);

private:

        float   _diameter;
        bool    _bShowName;
        int     _pointState;
        int     _pointGeometry;
        bool    _highlight;
        bool    _drawCenter;

        QGLWidget *_widget;

        bool     _bEpipolar;
        QPointF  _epipolar1;
        QPointF  _epipolar2;
};

class cCircle : public cObjectGL
{
    public:
        cCircle(QVector3D, QColor, float, float, bool, int dim);
        cCircle(int dim);

        void    draw();

    private:

        int     _dim; //plane in which circle is to be drawn (0=YZ, 1=XZ, 2=XY)
};

class cCross : public cObjectGL
{
    public:
        cCross(QVector3D, QColor, float, float, bool, int dim);

        void    draw();

    private:
        int     _dim;
};

class cBall : public cObjectGL
{
    public:

        cBall(QVector3D pt = QVector3D(0.f,0.f,0.f), float scale = 1.f, bool isVis = true, float lineWidth = 1.f);
        ~cBall();

        void    setPosition(QVector3D const &aPt);
        QVector3D   getPosition();
        void    setVisible(bool aVis);
        void    setScale(QVector3D aScale);

        void    draw();

        void setScale(float aScale);
private:

        cCircle *_cl0;
        cCircle *_cl1;
        cCircle *_cl2;

        cCross  *_cr0;
        cCross  *_cr1;
        cCross  *_cr2;
};

class cAxis : public cObjectGL
{
    public:
        cAxis(QVector3D pt = QVector3D(0.f,0.f,0.f), float scale = 1.f, float lineWidth = 1.f);

        void    draw();
};

class cGrid : public cObjectGL
{
    public:
        cGrid(QVector3D pt = QVector3D(0.f,0.f,0.f), QVector3D scale = QVector3D(1.f,1.f,1.f));

        void    draw();
};

class cBBox : public cObjectGL
{
    public:
        cBBox(QVector3D pt = QVector3D(0.f,0.f,0.f), QVector3D min= QVector3D(0.f,0.f,0.f), QVector3D max= QVector3D(1.f,1.f,1.f), float lineWidth = 1.f);

        void    draw();

        void set(QVector3D min, QVector3D max);

    private:
        QVector3D   _min;
        QVector3D   _max;
};

class cCamHandler
{
public:
    virtual ~cCamHandler(){}

    virtual void getCoins(QVector3D &aP1,QVector3D &aP2,QVector3D &aP3,QVector3D &aP4, double aZ) = 0;

    virtual QVector3D getCenter() = 0;

	virtual QVector3D getRotation() = 0;
};


class cCamGL : public cObjectGL
{
    public:
        cCamGL(cCamHandler *pCam, float scale, object_state state = state_default, float lineWidth = 1.f);

        void    draw();

        void    setpointSize(float size) { _pointSize = size; }

    private:
        float   _pointSize;

        cCamHandler *_Cam;
};

class cPolygonHelper;

class cMaskedImageGL;

class cPolygon : public cObjectGL
{
    public:

        cPolygon(int maxSz = INT_MAX, float lineWidth = 1.0f, QColor lineColor = Qt::green, QColor pointColor = Qt::red, int geometry = Geom_simple_circle, int style = LINE_NOSTIPPLE);

        ~cPolygon();


        virtual void draw();

        void    RemoveLastPointAndClose();

        void    close();

        bool    isPointInsidePoly(const QPointF& P);

        cPoint* findNearestPoint(const QPointF &pos, float getRadius = _selectionRadius);

        void    removeNearestOrClose(QPointF pos); //remove nearest point, or close polygon
        void    removeSelectedPoint();

        QString getSelectedPointName();
        int     getSelectedPointState();
        int     getSelectedPointGeometry();

        int     getSelectedPointIndex(){ return _idx; }

        void    setPointSelected();
        bool    isPointSelected(){ return _bSelectedPoint; }
        void    resetSelectedPoint();

        int     selectPoint(QString namePt);
        void    selectPoint(int idx);

        void    setPointSize(float size);
        float   getPointDiameter() { return _pointDiameter; }

        void    add(cPoint &pt);
        void    add(QPointF const &pt, bool selected=false, cPoint* lock = NULL);
        virtual void    addPoint(QPointF const &pt, cPoint* lock = NULL) ;

        void    clear();

        void    setClosed(bool closed) { _bIsClosed = closed; }
        bool    isClosed(){ return _bIsClosed; }

        int     size(){ return _points.size(); }

        cPoint & operator[](int ak){ return point(ak); }

        cPoint & point(int ak){ return _points[ak]; }

        const cPoint & operator[](int ak) const { return _points[ak]; }

        cPolygon & operator = (const cPolygon &);

        void    insertPoint( int i, const QPointF & value );

        void    insertPoint();

        void    removePoint( int i );

        QVector <QPointF> const getVector();
        QVector <QPointF> const getImgCoordVector(const cMaskedImageGL &img);
        QVector <QPointF> const transfoTerrain(const cMaskedImageGL &img);

        void    setVector(QVector <cPoint> const &aPts){ _points = aPts; }
        void    setVector(QVector <QPointF> const &aPts);

        cPolygonHelper* helper() { return _helper; }
        void    setHelper(cPolygonHelper* aHelper);

        virtual void refreshHelper(QPointF pos, bool insertMode, float zoom, bool ptIsVisible = true, cPoint* lock = NULL);

        int     finalMovePoint(cPoint* lock = NULL);

        void    removeLastPoint();

        // Points name
        void    showNames(bool show);
        bool    bShowNames() { return _bShowNames; }

        void    setDefaultName(QString name)    { _defPtName = name; }
        QString getDefaultName()                { return _defPtName; }

        void    rename(QPointF pos, QString name);

        void    showLines(bool show = true);
        bool    isLinear() { return _bShowLines; }

        void    translate(QPointF Tr);

        cPoint  translateSelectedPoint(QPointF Tr);

        float   getRadius()             { return _selectionRadius; }

        void    setRadius(float val)    { _selectionRadius = val;  }

        void    setParams(cParameters* aParams);

        float   getShiftStep()          { return _shiftStep; }

        void    setShiftStep(float val) { _shiftStep = val;  }

        bool    pointValid();

        void    setStyle(int style)     { _style = style; }

        void    setLineColor(QColor col){ _lineColor = col; }

        void    setMaxSize(int aMax)    { _maxSz = aMax; }

        int     getMaxSize()            { return _maxSz; }

        void    normalize(bool aBool)   { _bNormalize = aBool; }

        void	setAllVisible(bool visible);

        float   length();

    protected:

        cPolygon(int nbMax, float lineWidth, QColor lineColor,  QColor pointColor, bool withHelper, int geometry = Geom_simple_circle, int style = LINE_STIPPLE);

        QVector <cPoint>    _points;

        cPolygonHelper*     _helper;

        QColor              _lineColor;

        int                 _idx;

        int                 _style;

    private:

        float               _pointDiameter;

        int                 _pointGeometry;

        static float        _selectionRadius;

        //!states if polygon is closed
        bool                _bIsClosed;

        //!states if point with index _idx is selected
        bool                _bSelectedPoint;

        //!states if segments should be displayed
        bool                _bShowLines;

        //!states if names should be displayed
        bool                _bShowNames;

        QVector<qreal>      _dashes;
        QString             _defPtName;

        float               _shiftStep;

        //!vector max size
        int                 _maxSz;

        //!should image coordinates be normalized
        bool                _bNormalize;
};

class cPolygonHelper : public cPolygon
{
    public:

        cPolygonHelper(cPolygon* polygon, int nbMax, float lineWidth = 1.0f, QColor lineColor = Qt::blue, QColor pointColor = Qt::blue, int pointGeometry=Geom_simple_circle);

        ~cPolygonHelper();

        void   build(const cPoint &pos, bool insertMode);

        void   setPoints(cPoint p1, cPoint p2, cPoint p3);

    private:

        cPolygon* _polygon;
};

class cRectangle : public cPolygon
{
    public:

        cRectangle(int nbMax = 4, float lineWidth = 1.0f, QColor lineColor = Qt::green, int style = LINE_NOSTIPPLE);

        void    addPoint(QPointF const &pt, cPoint* lock = NULL);

        void    refreshHelper(QPointF pos, bool insertMode, float zoom, bool ptIsVisible = false, cPoint* lock = NULL);

        void    draw();
};

class cImageGL : public cObjectGL
{
    public:

        cImageGL(float gamma = 1.f);
        ~cImageGL();

        void    draw(QColor color);

        void    drawQuad(QColor color);

        static  void    drawQuad(GLfloat originX, GLfloat originY, GLfloat glw, GLfloat glh, QColor color = Qt::white);

        void    draw();

        void    setSize(QSize size);
        QSize   getSize() { return _size; }

#ifdef USE_MIPMAP_HANDLER
        void createTexture( MipmapHandler::Mipmap &aImage );
        void ImageToTexture( MipmapHandler::Mipmap &aImage );
        bool writeTiff( const std::string &aFilename ) const;
#else
        void    createTexture(QImage *pImg);

        void    ImageToTexture(QImage *pImg);
#endif
        void    deleteTexture();

        GLuint* getTexture(){return &_texture;}

        //height and width of original data
        int     width()  {return _size.width();}
        int     height() {return _size.height();}

        bool    isPtInside(QPointF const &pt);

        void    setGamma(float gamma){_gamma = (gamma >= 0) ? gamma : 0;}

        float   getGamma(){ return _gamma;}

        void    incGamma(float dgamma){setGamma(_gamma + dgamma);}

        static  void drawGradientBackground(int w,int h,QColor c1,QColor c2);

private:

        QGLShaderProgram _program;

        int     _texLocation;
        int     _gammaLocation;

        QSize   _size;

        //! Texture image
        GLuint  _texture;

        float   _gamma;
};

template<class T>
class cMaskedImage : public cObject
{

public:

    cMaskedImage(float gamma = 1.f, float sFactor= 1.f):
        _m_image(NULL),
        _m_mask(NULL),
        _m_rescaled_image(NULL),
        _m_rescaled_mask(NULL),
        _m_newMask(true),
        _gamma(gamma),
        _loadedImageRescaleFactor(sFactor),
        _loading(false)
    {

    }

    ~cMaskedImage()
    {
        deallocImages();
    }

    void deallocImages()
    {
        if(_m_image != NULL)
        {
            delete _m_image;
            _m_image = NULL;
        }
        if(_m_mask != NULL)
        {
            delete _m_mask;
            _m_mask = NULL;
        }
        if(_m_rescaled_image != NULL)
        {
            delete _m_rescaled_image;
            _m_rescaled_image = NULL;
        }
        if(_m_rescaled_mask != NULL)
        {
            delete _m_rescaled_mask;
            _m_rescaled_mask = NULL;
        }
    }

    T           *_m_image;
    T           *_m_mask;

    T           *_m_rescaled_image;
    T           *_m_rescaled_mask;

    bool        _m_newMask;
    float       _gamma;
    float       _loadedImageRescaleFactor;


    QSize		_fullSize;
    //QImageReader *_imageReader;
    bool		_loading;

};

class cMaskedImageGL : public cMaskedImage<cImageGL>, virtual public cObjectGL
{

public:
	#ifdef USE_MIPMAP_HANDLER
		cMaskedImageGL( MipmapHandler::Mipmap *aSrcImage, MipmapHandler::Mipmap *aSrcMask );
	#else
		cMaskedImageGL():
		    _qMaskedImage(NULL)
		{}

		cMaskedImageGL(QMaskedImage *qMaskedImage);
	#endif

    cMaskedImageGL(const QRectF & aRect);

    ~cMaskedImageGL();

//    ~cMaskedImageGL()
//    {}

    /*void setScale(QVector3D aScale)
    {
        _m_image->setScale(aScale);
        _m_mask->setScale(aScale);
    }*/

    float getLoadedImageRescaleFactor() { return _loadedImageRescaleFactor; }

    void  showMask(bool show) { _m_mask->setVisible(show); }

    void  draw();

    void  deleteTextures();

    void  createTextures();

    void  createFullImageTexture();

    cImageGL *       glImage()  { return _m_image; }
    const cImageGL * glImage() const { return _m_image; }
    cImageGL*   glMask()   { return _m_mask;  }

    void		copyImage(cMaskedImage<QImage>* image, QRect& rect);

    QSize		fullSize();

	#ifdef USE_MIPMAP_HANDLER
		bool hasSrcImage() const { return mSrcImage != NULL; }

		MipmapHandler::Mipmap & srcImage()
		{
			ELISE_DEBUG_ERROR( !hasSrcImage(), "cMaskedImageGL::srcImage()", "!hasSrcImage");
			return *mSrcImage;
		}

		const MipmapHandler::Mipmap & srcImage() const
		{
			ELISE_DEBUG_ERROR( !hasSrcImage(), "cMaskedImageGL::srcImage()", "!hasSrcImage");
			return *mSrcImage;
		}

		bool hasSrcMask() const { return mSrcMask != NULL; }

		MipmapHandler::Mipmap & srcMask()
		{
			ELISE_DEBUG_ERROR( !hasSrcMask(), "cMaskedImageGL::srcMask()", "!hasSrcMask");
			return *mSrcMask;
		}

		void removeSrcMask() { mSrcMask = NULL; }
	#else
		bool           hasQImage() const { return _qMaskedImage != NULL; }
		QMaskedImage * getMaskedImage() { return _qMaskedImage; }
		void           setMaskedImage(QMaskedImage * aMaskedImage) { _qMaskedImage = aMaskedImage; }
	#endif
private:
	#ifdef USE_MIPMAP_HANDLER
		MipmapHandler::Mipmap *mSrcImage;
		MipmapHandler::Mipmap *mSrcMask;
	#else
		QMaskedImage *_qMaskedImage;
	#endif

    QMutex			_mutex;
};

//====================================================================================

//! Default message positions on screen
enum MessagePosition {  LOWER_LEFT_MESSAGE,
                        LOWER_RIGHT_MESSAGE,
                        LOWER_CENTER_MESSAGE,
                        UPPER_CENTER_MESSAGE,
                        SCREEN_CENTER_MESSAGE
};

//! Temporary Message to display
struct MessageToDisplay
{
    MessageToDisplay():
        color(Qt::white),
        position(LOWER_CENTER_MESSAGE)
    {}

    //! Message
    QString message;

    //! Color
    QColor color;

    //! Message position on screen
    MessagePosition position;
};

class cMessages2DGL : public cObjectGL
{
public:

    cMessages2DGL(QGLWidget *glw):
        _bDrawMessages(true),
        m_font(QFont("Arial", 10, QFont::Normal, false)),
        glwid(glw),
        w(0),
        h(0)
    {}

    void draw();

    int renderTextLine(MessageToDisplay messageTD, int x, int y, int sizeFont = 10);

    void displayNewMessage(const QString& message,
                                       MessagePosition pos = SCREEN_CENTER_MESSAGE,
                                       QColor color = Qt::white);

    void constructMessagesList(bool show, int mode, bool m_bDisplayMode2D, bool dataloaded, float zoom);

    std::list<MessageToDisplay>::iterator GetLastMessage();

    std::list<MessageToDisplay>::iterator GetPenultimateMessage();

    MessageToDisplay& LastMessage();

    void wh(int ww,int hh)
    {
        w=ww;
        h=hh;
    }

    bool drawMessages(){ return _bDrawMessages && size(); }

    void showMessages(bool show) { _bDrawMessages = show; }

    size_t size(){ return m_messagesToDisplay.size(); }

    void    glRenderText(QString text,QPointF pt,QColor color);

private:

    bool _bDrawMessages;

    list<MessageToDisplay> m_messagesToDisplay;

    //! Default font
    QFont m_font;

    QGLWidget *glwid;

    int w;
    int h;
};

void glDrawUnitCircle(uchar dim, float cx = 0.f, float cy = 0.f, float r = 1.f, int steps = 128);
void glDrawEllipse(float cx, float cy, float rx=3.f, float ry= 3.f, int steps = 64);

string eToString( QImage::Format e );
std::ostream & operator <<( std::ostream &aStream, const QSize &aSize );

#ifdef __DEBUG
	#define CHECK_GL_ERROR(where) __check_gl_error(where)

	string glErrorToString(GLenum aEnum);

	inline void __check_gl_error(const std::string &aWhere)
	{
		const GLenum err = glGetError();

		ELISE_DEBUG_ERROR(err != GL_NO_ERROR, aWhere, "glGetError() = " << glErrorToString(err));
	}
#else
	#define CHECK_GL_ERROR(where)
#endif

#endif //__3DObject__

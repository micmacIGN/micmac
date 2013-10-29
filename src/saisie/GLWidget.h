#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

#include <QtOpenGL/QGLWidget>
#include <QtOpenGL/QGLBuffer>
#include <QGLContext>
#include <QDebug>

#include "GL/glu.h"

#include <QUrl>
#include <QtGui/QMouseEvent>
#include <QSettings>
#include <QMimeData>
#include <QTime>
#include <QPainter>


#include "Data.h"
#include "Engine.h"
#include "3DTools.h"
#include "mainwindow.h"

#include "3DObject.h"

//! View orientation
enum VIEW_ORIENTATION {  TOP_VIEW,      /**< Top view (eye: +Z) **/
                         BOTTOM_VIEW,	/**< Bottom view **/
                         FRONT_VIEW,	/**< Front view **/
                         BACK_VIEW,     /**< Back view **/
                         LEFT_VIEW,     /**< Left view **/
                         RIGHT_VIEW     /**< Right view **/
};

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:

    //! Default constructor
    GLWidget(QWidget *parent = NULL, cData* data = NULL);

    //! Destructor
    ~GLWidget();

    bool eventFilter(QObject* object, QEvent* event);

    //! Set data to display
    void setData(cData* data);

    cData* getData() {return m_Data;}

    //! Interaction mode (only in 3D)
    enum INTERACTION_MODE { TRANSFORM_CAMERA,
                            SELECTION
    };

    //! Default message positions on screen
    enum MessagePosition {  LOWER_LEFT_MESSAGE,
                            LOWER_CENTER_MESSAGE,
                            UPPER_CENTER_MESSAGE,
                            SCREEN_CENTER_MESSAGE
    };

    //! Displays a status message
    /** \param message message (if message is empty, all messages will be cleared)
        \param pos message position on screen
    **/
    virtual void displayNewMessage(const QString& message,
                                   MessagePosition pos = SCREEN_CENTER_MESSAGE);

    //! States if data (cloud, camera or image) is loaded
    bool hasDataLoaded(){return m_Data->NbClouds()||m_Data->NbCameras() ||m_Data->NbImages();}

    //! Sets camera to a predefined view (top, bottom, etc.)
    void setView(VIEW_ORIENTATION orientation);

    //! Sets current zoom
    void setZoom(float value);

    void zoomFit();

    void zoomFactor(int percent);

    //! Switch between move mode and selection mode (only in 3D)
    void setInteractionMode(INTERACTION_MODE mode);

    //! Shows axis or not
    void showAxis(bool show);

    //! Shows ball or not
    void showBall(bool show);

    //! Shows cams or not
    void showCams(bool show);

    //! Shows help messages or not
    void showMessages(bool show);

    //! Shows bounding box or not
    void showBBox(bool show);

    //! States if help messages should be displayed
    bool showMessages();

    //! Display help messages for selection mode
    void displaySelectionMessages();

    //! Display help messages for move mode
    void displayMoveMessages();

    //! Select points with polyline
    void Select(int mode);

    //! Delete current polyline
    void clearPolyline();

     //! Undo all past selection actions
    void undoAll();

    //! Increase or decrease point size
    void ptSizeUp(bool);

    void setBufferGl(bool onlyColor = false);

    void getProjection(QPointF &P2D, Vertex P);

    QVector <selectInfos> getSelectInfos(){return m_infos;}

    //! Avoid all past actions
    void reset();

    //! Reset view
    void resetView();

    void resetRotationMatrix();

    void resetTranslationMatrix();

    QImage* getGLMask(){return _mask;}

    ViewportParameters* getParams(){return &m_params;}

    void applyGamma(float aGamma);

    void drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw);

    void drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw, QColor color);

    void drawQuad(GLfloat originX, GLfloat originY, GLfloat glh, GLfloat glw, GLuint idTexture);

    void enableOptionLine();
    void disableOptionLine();

public slots:
    void zoom();

    //! called when receiving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

    void selectedPoint(uint idCloud, uint idVertex,bool selected);

    void interactionMode(bool modeSelection);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mouseDoubleClickEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent* event);

    void ImageToTexture(GLuint idTexture,QImage* image);

    //! Initialization state of GL
    bool m_bGLInitialized;

    //inherited from QWidget (drag & drop support)
    virtual void dragEnterEvent(QDragEnterEvent* event);
    virtual void dropEvent(QDropEvent* event);

    //! Draw widget gradient background
    void drawGradientBackground();
    
    //! Draw selection polygon
    void drawPolygon();

    //! Draw one point and two segments (for insertion or move)
    void drawPointAndSegments();

    void drawPointAndSegments(const QVector<QPointF> &aPoly);

    //! Fill m_polygon2 for point insertion or move
    void fillPolygon2(const QPointF &pos);

    //! set index of cursor closest point
    void findClosestPoint(const QPointF &pos, float sqr_radius);

    //! GL context aspect ratio m_glWidth/m_glHeight
    float m_glRatio;

    //! ratio between GL context size and image size
    float m_rw, m_rh;

    //! Default font
    QFont m_font;

    //! States if messages should be displayed
    bool m_bDrawMessages;

    //! States if view is centered on object
    bool m_bObjectCenteredView;

    //! Current interaction mode (with mouse)
    INTERACTION_MODE m_interactionMode;

    bool m_bFirstAction;

    //! Temporary Message to display
    struct MessageToDisplay
    {
        //! Message
        QString message;
        //! Message position on screen
        MessagePosition position;
    };

    //! Texture image
    GLuint m_textureImage;

    GLuint m_textureMask;

    //! List of messages to display
    list<MessageToDisplay> m_messagesToDisplay;

    QString m_messageFPS;

    //! Point list for polygonal selection
    cPolygon    m_polygon;

    //! Point list for polygonal insertion
    QVector < QPointF > m_polygon2;

    //! Viewport parameters (zoom, etc.)
    ViewportParameters m_params;

    //! Data to display
    cData      *m_Data;

    //! acceleration factor
    float       m_speed;

    //! selection infos stack
    QVector <selectInfos> m_infos;

    //! states if display is 2D or 3D
    bool        m_bDisplayMode2D;

    //! data position in the gl viewport
    GLfloat     m_glPosition[2];

    //! click counter to manage point move event
    int         m_Click;

    //! (square) radius for point selection
    float       m_sqr_radius;

    QPointF WindowToImage(const QPointF &pt);

    QPointF ImageToWindow(const QPointF &im);

    QPointF     m_lastMoveImg;
    QPointF     m_lastClickWin;
    QPointF     m_lastClickZoom;

    QPointF     m_lastPos;

private:

    void        setProjectionMatrix();
    void        computeFPS();

    QGLBuffer   _vertexbuffer;
    QGLBuffer   _vertexColor;

    int         _frameCount;
    int         _previousTime;
    int         _currentTime;

    int         _idx;

    float       _fps;

    bool        _g_mouseLeftDown;
    bool        _g_mouseMiddleDown;
    bool        _g_mouseRightDown;
    GLfloat     _g_tmpoMatrix[9];
    GLfloat     _g_rotationOx[9];
    GLfloat     _g_rotationOy[9];
    GLfloat     _g_rotationOz[9];
    GLfloat     _g_rotationMatrix[9];
    GLfloat     _g_glMatrix[16];
    QTime       _time;

    QImage      _glImg;
    QImage      *_mask;
    GLdouble    *_mvmatrix;
    GLdouble    *_projmatrix;
    GLint       *_glViewport;

    cBall       *_theBall;
    cAxis       *_theAxis;
    cBBox       *_theBBox;

    QVector < cCam* > _pCams;
};

#endif  /* _GLWIDGET_H */


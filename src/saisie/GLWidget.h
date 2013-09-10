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

    //! Set data to display
    void setData(cData* data);

    //! Interaction mode (with the mouse!)
    enum INTERACTION_MODE { TRANSFORM_CAMERA,
                            SELECTION
    };

    //! Default message positions on screen
    enum MessagePosition {  LOWER_LEFT_MESSAGE,
                            LOWER_CENTER_MESSAGE,
                            UPPER_CENTER_MESSAGE,
                            SCREEN_CENTER_MESSAGE
    };

    void    setSelectionMode(int mode ) {_m_selection_mode = mode; }
    int     getSelectionMode()          {return _m_selection_mode;}

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

    //! Switch between move mode and selection mode
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
    void showSelectionMessages();

    //! Display help messages for move mode
    void showMoveMessages();

    //! Select points with polyline
    void Select(int mode);

    //! Insert point in polyline
    void insertPolylinePoint();

    //! Delete mouse closest point
    void deletePolylinePoint();

    //! Delete current polyline
    void clearPolyline();

    //! Close polyline
    void closePolyline();

     //! Undo all past selection actions
    void undoAll();

    //! Increase or decrease point size
    void ptSizeUp(bool);

    void setBufferGl(bool onlyColor = false);

    void getProjection(QPointF &P2D, Vertex P);

    QVector <selectInfos> getSelectInfos(){return m_infos;}

    void reset();

    void WindowToImage(QPointF const &p0, QPointF &p1);

    QImage* getGLImage(){return &_glImg;}

public slots:
    void zoom();

    //! called when receiving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

    //! Signal emitted when the mouse wheel is rotated
    void mouseWheelRotated(float wheelDelta_deg);

    void selectedPoint(uint idCloud, uint idVertex,bool selected);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent* event);

    //! Initialization state of GL
    bool m_bGLInitialized;

    //inherited from QWidget (drag & drop support)
    virtual void dragEnterEvent(QDragEnterEvent* event);
    virtual void dropEvent(QDropEvent* event);

    //! Draw frame axis
    void drawAxis();

    //! Draw ball
    void drawBall();

    //! Draw ball
    void drawCams();

    //! Draw bounding box
    void drawBbox();

    //! Draw widget gradient background
    void drawGradientBackground();

    void setStandardOrthoCenter();

    GLuint getNbGLLists() { return m_nbGLLists; }
    void incrNbGLLists() { m_nbGLLists++; }
    void resetNbGLLists(){ m_nbGLLists = 0; }

    //! GL context width
    int m_glWidth;
    //! GL context height
    int m_glHeight;

    //! ratio between GL context size and image size
    float m_rw, m_rh;

    //! Default font
    QFont m_font;

    //! States if frame axis should be drawn
    bool m_bDrawAxis;

    //! States if ball should be drawn
    bool m_bDrawBall;

    //! States if cams should be drawn
    bool m_bDrawCams;

    //! States if messages should be displayed
    bool m_bDrawMessages;

    //! States if Bounding box should be displayed
    bool m_bDrawBbox;

    //! States if view is centered on object
    bool m_bObjectCenteredView;

    //! States if selection polyline is closed
    bool m_bPolyIsClosed;

    //! Current interaction mode (with mouse)
    INTERACTION_MODE m_interactionMode;

    bool m_bFirstAction;

    int m_previousAction;

    //! Temporary Message to display
    struct MessageToDisplay
    {
        //! Message
        QString message;
        //! Message position on screen
        MessagePosition position;
    };

    //! Trihedron GL list
    GLuint m_trihedronGLList;

    //! Ball GL list
    GLuint m_ballGLList;

    //! Texture GL list
    GLuint m_texturGLList;

    int m_nbGLLists;

    //! List of messages to display
    list<MessageToDisplay> m_messagesToDisplay;

    QString m_messageFPS;

    //! Point list for polygonal selection
    QVector < QPoint > m_polygon;

    //! Viewport parameters (zoom, etc.)
    ViewportParameters m_params;

    //! Data to display
    cData *m_Data;

    //! acceleration factor
    float m_speed;

    //! selection infos stack
    QVector <selectInfos> m_infos;

    //! states if display is 2D or 3D
    bool m_bDisplayMode2D;

    //! data position in the gl viewport
    GLfloat m_glPosition[2];

    //! transparency of deleted areas
    float   m_alpha;

private:

    QPoint      m_lastPos;

    void        setProjectionMatrix();
    void        computeFPS();

    QGLBuffer   m_vertexbuffer;
    QGLBuffer   m_vertexColor;

    int         _frameCount;
    int         _previousTime;
    int         _currentTime;

    float       _fps;

    int         _m_selection_mode;

    double      _MM[16];
    double      _MP[16];
    int         _VP[4];

    bool        _m_g_mouseLeftDown;
    bool        _m_g_mouseMiddleDown;
    bool        _m_g_mouseRightDown;
    GLfloat     _m_g_tmpoMatrix[9];
    GLfloat     _m_g_rotationOx[9];
    GLfloat     _m_g_rotationOy[9];
    GLfloat     _m_g_rotationOz[9];
    GLfloat     _m_g_rotationMatrix[9];
    GLfloat     _m_g_glMatrix[16];
    QTime       _time;

    QImage      _glImg;
};

#endif  /* _GLWIDGET_H */


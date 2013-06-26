#ifndef _GLWIDGET_H
#define _GLWIDGET_H


#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>

#ifndef  WIN32    
    #include "GL/glew.h"
    #include "GL/glut.h"
#endif

#include <QtOpenGL/QGLWidget>
#include <QtOpenGL/QGLBuffer>
#include <QGLContext>

#ifdef  WIN32
    #include "GL/glu.h"
#endif

#include <QUrl>
#include <QtGui/QMouseEvent>
#include <QSettings>
#include <QMessageBox>
#include <QMimeData>

#include "Data.h"
#include "Engine.h"

//! View orientation
enum VIEW_ORIENTATION {  TOP_VIEW,      /**< Top view (eye: +Z) **/
                         BOTTOM_VIEW,	/**< Bottom view **/
                         FRONT_VIEW,	/**< Front view **/
                         BACK_VIEW,     /**< Back view **/
                         LEFT_VIEW,     /**< Left view **/
                         RIGHT_VIEW     /**< Right view **/
};

class ViewportParameters
{
public:
    //! Default constructor
    ViewportParameters();

    //! Copy constructor
    ViewportParameters(const ViewportParameters& params);

    //! Destructor
    ~ViewportParameters();

    //! Current zoom
    float zoom;

    //! Point size
    float PointSize;

    //! Line width
    float LineWidth;

    //! Rotation angles
    float angleX;
    float angleY;

    //! Translation matrix
    GLfloat m_translationMatrix[3];
};

class GLWidget : public QGLWidget
{
    Q_OBJECT

private:

    QPoint m_lastPos;

public:

    //! Default constructor
    GLWidget(QWidget *parent = NULL, cData* data = NULL);

    //! Destructor
    ~GLWidget();

    //! Set data to display
    void setData(cData* data);

    //! Interaction mode (with the mouse!)
    enum INTERACTION_MODE { TRANSFORM_CAMERA,
                            SEGMENT_POINTS
    };

    //! Default message positions on screen
    enum MessagePosition {  LOWER_LEFT_MESSAGE,
                            LOWER_CENTER_MESSAGE,
                            UPPER_CENTER_MESSAGE,
                            SCREEN_CENTER_MESSAGE
    };

    void    setSelectionMode(int mode ) {m_selection_mode = mode; }
    int     getSelectionMode()          {return m_selection_mode;}

    //! Displays a status message
    /** \param message message (if message is empty, all messages will be cleared)
        \param pos message position on screen
    **/
    virtual void displayNewMessage(const QString& message,
                                   MessagePosition pos = SCREEN_CENTER_MESSAGE);

    //! States if data (cloud or camera) is loaded
    bool hasDataLoaded(){return m_bCloudLoaded||m_bCameraLoaded;}

    //! States if a cloud is loaded
    bool hasCloudLoaded(){return m_bCloudLoaded;}

    //! Sets cloud state as loaded
    void setCloudLoaded(bool isLoaded) { m_bCloudLoaded = isLoaded; }

    //! Sets camera state as loaded
    void setCameraLoaded(bool isLoaded) { m_bCameraLoaded = isLoaded; }

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

    //! States if help messages should be displayed
    bool showMessages();

    //! Display help messages for selection mode
    void showSelectionMessages();

    //! Display help messages for move mode
    void showMoveMessages();

    //! Segment points with polyline
    void Select(int mode);

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

    //! Save viewing directions and polylines in Filename
    void saveSelectionInfos(QString Filename);

    void setBufferGl(bool onlyColor = false);

    void getProjection(Pt2df &P2D, Vertex P);
public slots:
    void zoom();

    //! called when receiving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

    //! Signal emitted when the mouse wheel is rotated
    void mouseWheelRotated(float wheelDelta_deg);

    void SelectedPoint(uint idCloud, uint idVertex,bool selected);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent* event);

    //! Initialization state
    bool m_bInitialized;

    //inherited from QWidget (drag & drop support)
    virtual void dragEnterEvent(QDragEnterEvent* event);
    virtual void dropEvent(QDropEvent* event);

    void draw3D();

    //! Draw frame axis
    void drawAxis();

    //! Draw ball
    void drawBall();

    //! Draw ball
    void drawCams();

    //! Draw widget gradient background
    void drawGradientBackground();

    void setStandardOrthoCenter();

    GLuint getNbGLLists() { return m_nbGLLists; }
    void incrNbGLLists() { m_nbGLLists++; }
    void resetNbGLLists(){ m_nbGLLists = 0; }

    void storeInfos(bool inside, bool add);

    void setAngles(float angleX, float angleY);

    //! GL context width
    int m_glWidth;
    //! GL context height
    int m_glHeight;

    //! Default font
    QFont m_font;

    //! States if a cloud is already loaded
    bool m_bCloudLoaded;

    //! States if a camera is already loaded
    bool m_bCameraLoaded;

    //! States if frame axis should be drawn
    bool m_bDrawAxis;

    //! States if ball should be drawn
    bool m_bDrawBall;

    //! States if cams should be drawn
    bool m_bDrawCams;

    //! States if messages should be displayed
    bool m_bMessages;

    //! States if view is centered on object
    bool m_bObjectCenteredView;

    //! States if selection polyline is closed
    bool m_bPolyIsClosed;

    //! Current interaction mode (with mouse)
    INTERACTION_MODE m_interactionMode;

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

    int m_nbGLLists;

    //! List of messages to display
    list<MessageToDisplay> m_messagesToDisplay;

    QString m_messageFPS;

    //! Point list for polygonal selection
    std::vector < Pt2df > m_polygon;

    //! Viewport parameters (zoom, etc.)
    ViewportParameters m_params;

    //! Input infos list
    QVector < cSelectInfos > m_infos;

    //! Data to display
    cData *m_Data;

    //! acceleration factor
    float m_speed;

private:

    void        setProjectionMatrix();
    void        calculateFPS();

    QGLBuffer   m_vertexbuffer;
    QGLBuffer   m_vertexColor;

    uint        _frameCount;
    uint        _previousTime;
    uint        _currentTime;

    float       _fps;

    int         m_selection_mode;

    double      _MM[16];
    double      _MP[16];
    int         _VP[4];

};

#endif  /* _GLWIDGET_H */


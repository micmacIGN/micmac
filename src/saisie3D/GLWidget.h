#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <QtOpenGL/QGLWidget>
#include <QGLContext>
#include <QUrl>

#include "Data.h"

//! View orientation
enum VIEW_ORIENTATION {  TOP_VIEW,	/**< Top view (eye: +Z) **/
                         BOTTOM_VIEW,	/**< Bottom view **/
                         FRONT_VIEW,	/**< Front view **/
                         BACK_VIEW,	/**< Back view **/
                         LEFT_VIEW,	/**< Left view **/
                         RIGHT_VIEW	/**< Right view **/
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
};

class GLWidget : public QGLWidget
{
private:
    //QVector <Cloud_::Cloud> m_ply;

    Q_OBJECT

    QPoint              m_lastPos;

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
                            UPPER_CENTER_MESSAGE,
                            SCREEN_CENTER_MESSAGE
    };

    //! Message type
    enum MessageType {  CUSTOM_MESSAGE,                       
                        MANUAL_SEGMENTATION_MESSAGE
    };

    //! Displays a status message in the bottom-left corner
    /** WARNING: currently, 'append' is not supported for SCREEN_CENTER_MESSAGE
        \param message message (if message is empty and append is 'false', all messages will be cleared)
        \param pos message position on screen
        \param type message type (if not custom, only one message of this type at a time is accepted)
    **/
    virtual void displayNewMessage(const QString& message,
                                   MessagePosition pos,
                                   MessageType type=CUSTOM_MESSAGE);

    //! States if a cloud is loaded
    bool hasCloudLoaded(){return m_bCloudLoaded;}

    //! Sets cloud state as loaded
    void setCloudLoaded(bool isLoaded) { m_bCloudLoaded = isLoaded; }

    //! Sets camera to a predefined view (top, bottom, etc.)
    void setView(VIEW_ORIENTATION orientation);

    //! Updates current zoom
    void updateZoom(float zoomFactor);

    //! Sets current zoom
    void setZoom(float value);

    void setInteractionMode(INTERACTION_MODE mode);

     //! Shows ball or not
    void showBall(bool show);

    //! Segment points with polyline
    void segment(bool inside);

    //! Delete current polyline
    void clearPolyline();

    //! Close polyline
    void closePolyline();

     //! Undo all past selection actions
    void undoAll();

    //! Increase or decrease point size
    void ptSizeUp(bool);

public slots:
    void zoom();

    //! called when receiving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames);

    //! Signal emitted when the mouse wheel is rotated
    void mouseWheelRotated(float wheelDelta_deg);

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

    void drawGradientBackground();

    void setStandardOrthoCenter();

    //! GL context width
    int m_glWidth;
    //! GL context height
    int m_glHeight;

    //! Default font
    QFont m_font;

    //! States if a cloud is already loaded
    bool m_bCloudLoaded;

    //! States if frame axis should be drawn
    bool m_bDrawAxis;

    //! States if ball should be drawn
    bool m_bDrawBall;

    //! States if view is centered on object
    bool m_bObjectCenteredView;

    //! States if selection polyline is closed
    bool m_bPolyIsClosed;

    //! Current interaction mode (with mouse)
    INTERACTION_MODE m_interactionMode;

    //! Temporary Message to display in the lower-left corner
    struct MessageToDisplay
    {
        //! Message
        QString message;
        //! Message position on screen
        MessagePosition position;
        //! Message type
        MessageType type;
    };

    //! Trihedron GL list
    GLuint m_trihedronGLList;

    //! Ball GL list
    GLuint m_pivotGLList;

    //! List of messages to display
    list<MessageToDisplay> m_messagesToDisplay;

    //! Point list for polygonal selection
    QVector < QPoint > m_polygon;

    //! Viewport parameters (zoom, etc.)
    ViewportParameters m_params;

    //! Data to display
    cData *m_Data;
};

#endif  /* _GLWIDGET_H */


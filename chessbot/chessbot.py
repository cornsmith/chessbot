# -*- coding: utf-8 -*-
"""
Chessbot is a real-time chess assistant using:
    - computer vision (OpenCV)
    - chess library (python-chess)
    - chess engine (Stockfish)
"""

import pickle
import argparse
import cv2
import numpy as np
import chess

from chess import polyglot
from chess import uci
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


CHESSBOARD_SQUARES = 64
CHESSBOARD_WIDTH = 8
SQUARE_WIDTH = 100
CV_CAP_PROP_FRAME_WIDTH = 1280
CV_CAP_PROP_FRAME_HEIGHT = 720
UI_WINDOW_WIDTH = UI_WINDOW_HEIGHT = 500


def get_single_image():
    # initialise camera, read an image, and close
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CV_CAP_PROP_FRAME_HEIGHT)
    _, img = cap.read()
    cap.release()
    return img


def project_board(img, corners):
    """
    Takes a raw image and projects the squares within the cropped board.
    Input:
        img: multi-channel image, 3D array
        corners: corners from get_chessboard_corners
    Output:
        board_img: RGB image
    """

    def impute_corners(corners):
        """
        Since findChessboardCorners only detects the inside corners, this function
        resizes (w+2, h+2) and imputes the extra corners.
        Assumes w == h.
        """
        w = CHESSBOARD_WIDTH - 1

        # new array
        new_cnr = np.zeros((w+2, w+2, 2), np.float32)

        # inside edges
        new_cnr[1:w+1, 1:w+1, :] = corners.reshape(w, w, 2)

        # outside edges
        new_cnr[0, :, :] = new_cnr[1, :, :] - (new_cnr[2, :, :] - new_cnr[1, :, :])
        new_cnr[w+1, :, :] = new_cnr[w, :, :] + (new_cnr[w, :, :] - new_cnr[w-1, :, :])
        new_cnr[:, 0, :] = new_cnr[:, 1, :] - (new_cnr[:, 2, :] - new_cnr[:, 1, :])
        new_cnr[:, w+1, :] = new_cnr[:, w, :] + (new_cnr[:, w, :] - new_cnr[:, w-1, :])

        return new_cnr

    def project_square(img, x, y):
        """
        4-Point deskew using Perspective Transform
        """
        pts1 = np.float32([
            corners[x, y],
            corners[x, y+1],
            corners[x+1, y],
            corners[x+1, y+1],
        ])
        pts2 = np.float32([
            [0, 0],
            [SQUARE_WIDTH, 0],
            [0, SQUARE_WIDTH],
            [SQUARE_WIDTH, SQUARE_WIDTH],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        sq_img = cv2.warpPerspective(img, M, (SQUARE_WIDTH, SQUARE_WIDTH))

        return sq_img

    # impute outer corners
    corners = impute_corners(corners)

    # dimensions of board image
    w = h = SQUARE_WIDTH * CHESSBOARD_WIDTH
    board_img = np.zeros((w, h, img.shape[2]), np.uint8)

    for x in range(CHESSBOARD_WIDTH):
        for y in range(CHESSBOARD_WIDTH):
            board_img[
                (x * SQUARE_WIDTH):((x + 1) * SQUARE_WIDTH),
                (y * SQUARE_WIDTH):((y + 1) * SQUARE_WIDTH),
                :
            ] = project_square(img, x, y)

    return board_img


def get_square_image(board_img, sq):
    """
    Outputs the square image based on square index {0...63}
    """
    x_offset = CHESSBOARD_WIDTH - 1 - int(sq / CHESSBOARD_WIDTH)
    y_offset = (sq % CHESSBOARD_WIDTH)
    x = x_offset * SQUARE_WIDTH
    y = y_offset * SQUARE_WIDTH

    return board_img[x:x+SQUARE_WIDTH, y:y+SQUARE_WIDTH, :]


def radial_gray_hist(sq_img):
    """
    Apply centre-weighted radial kernel and calculate grayscale histogram
    """
    # must be a square
    h, w, _ = sq_img.shape
    if h != w:
        raise Exception('Square image does not have h = w')

    # centre radial weights
    rad_mat = np.zeros((h, w), np.float32)
    for x in range(h):
        for y in range(w):
            rad_mat[x, y] = ((h - abs(h - x - x - 1)) / h) * ((w - abs(w - y - y - 1)) / w)

    # make everything outside radius / 3 -> 0 weight
    circle_img = np.zeros((h, w), np.uint8)
    cv2.circle(circle_img, (int(w / 2), int(h / 2)), int(SQUARE_WIDTH / 3), 1, thickness=-1)
    rad_mat = cv2.bitwise_and(rad_mat, rad_mat, mask=circle_img)

    # calculated grayscale histogram and gray value mode
    sq_img = cv2.cvtColor(sq_img, cv2.COLOR_BGR2GRAY)
    hist = [rad_mat.reshape(h*w)[sq_img.reshape(h*w) == group].sum() for group in range(256)]

    return hist


def square_has_piece(sq_img):
    """
    Returns True / False on whether a square image has a piece in it (any colour)
    """
    def piece_edge_scores(sq_img):
        """
        Returns a list of scores + a masked image used in scores from a square image
        """
        def get_masked_circle(img):
            try:
                h, w, _ = img.shape
            except:
                h, w = img.shape
            circle_img = np.zeros((h, w), np.uint8)
            cv2.circle(circle_img, (int(w / 2), int(h / 2)), int(SQUARE_WIDTH / 3), 1, thickness=-1)
            masked_img = cv2.bitwise_and(img, img, mask=circle_img)
            return masked_img

        def apply_threshold(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return img

        # create copies with transformations / masks
        masked = get_masked_circle(sq_img)
        bw_sq_img = apply_threshold(sq_img)
        bw_m_sq_img = get_masked_circle(bw_sq_img)

        # edge score 1 - circle masked RGB
        edges1 = cv2.Canny(masked, 50, 100, 3)
        e1_score = int(np.mean(edges1))

        # edge score 2 - circle masked BW
        edges2 = cv2.Canny(bw_m_sq_img, 50, 100, 3)
        e2_score = int(np.mean(edges2))

        # edge score 1 - circle masked RGB
        edges1 = cv2.Canny(masked, 100, 150, 3)
        e3_score = int(np.mean(edges1))

        # edge score 2 - circle masked BW
        edges2 = cv2.Canny(bw_m_sq_img, 100, 150, 3)
        e4_score = int(np.mean(edges2))

        # contour score 1 - circle masked BW
        _, contours, _ = cv2.findContours(bw_m_sq_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c1_score = sum([len(x.flatten()) for x in contours])

        return [e1_score, e2_score, e3_score, e4_score, c1_score], masked

    # TODO: turn this into a model instead of hard-coding parameters
    scores, x_img = piece_edge_scores(sq_img)
    has_piece = (scores[0] > 6) + (scores[1] > 6) + (scores[2] > 6) + (scores[3] > 6) + (scores[4] > 300) > 2

    return has_piece, scores, x_img


def fit_piece_colours(board_img):
    """
    Build model for detecting whether a square has {
        1 - player 1 piece
        2 - player 2 piece
        0 - no piece
    }
    Assumes board_img is an calibrated image at starting position
    """
    # default calibration layout
    #labels = [x for x in '1' * 16 + '0' * 32 + '2' * 16]
    labels1 = labels2 = [x for x in '1' * 8 + '2' * 8]

    # define feature signature for each square
    X1 = []
    X2 = []
    for sq in range(CHESSBOARD_SQUARES):
        # get feature signature for square
        sq_img = get_square_image(board_img, sq)
        hist = radial_gray_hist(sq_img)
        cv2.imwrite('./calibration/sq_{0}.jpg'.format(sq), sq_img)

        # prepare 2 datasets for alternating squares
        if not 15 < sq < 48:
            if sq % 2 == (sq // 8) % 2:
                X1.append(hist)
            else:
                X2.append(hist)

    # Fit NN for 2 models only for piece colour, excludes no-piece
    # TODO: tune k instead of hard-code k
    neigh1 = KNeighborsClassifier(n_neighbors=3)
    neigh1.fit(X1, labels1)
    neigh2 = KNeighborsClassifier(n_neighbors=3)
    neigh2.fit(X2, labels2)

    # Measure performance - cheat and show predicted values on training set
    squares = []
    for sq in range(CHESSBOARD_SQUARES):
        sq_img = get_square_image(board_img, sq)
        hist = radial_gray_hist(sq_img)
        hist = np.array(hist).reshape(1, -1)
        shp, ss, test_img = square_has_piece(sq_img)
        print(sq, ss)
        cv2.imwrite('./calibration/m_{0}.jpg'.format(sq), test_img)
        if shp:
            if sq % 2 == (sq // 8) % 2:
                squares.append(neigh1.predict(hist)[0])
            else:
                squares.append(neigh2.predict(hist)[0])
        else:
            squares.append('0')

    print('Current predicted values', ''.join(squares))

    return (neigh1, neigh2)


def format_squares_piece_text(sq_pc):
    i = 0
    sq_pc_formatted = []
    for pc in sq_pc:
        i += 1
        sq_pc_formatted.append(pc)
        sq_pc_formatted.append(' ')
        if i % 8 == 0:
            sq_pc_formatted.append('\n')

    return ''.join(sq_pc_formatted)


def get_piece_colours(board_img, board=None, meth='model'):
    '''
    Predict piece_model on every square of input board image
    Output:
        64 character string of {0,1,2}
    '''
    squares = []

    if meth == 'model':
        neigh1, neigh2 = piece_model

        for sq in range(CHESSBOARD_SQUARES):
            # get feature signature for square
            sq_img = get_square_image(board_img, sq)
            hist = radial_gray_hist(sq_img)

            # predict {1, 2} based on piece model built at calibration
            hist = np.array(hist).reshape(1, -1)

            shp, _, _ = square_has_piece(sq_img)
            if shp:
                if sq % 2 == (sq // 8) % 2:
                    squares.append(neigh1.predict(hist)[0])
                else:
                    squares.append(neigh2.predict(hist)[0])
            else:
                squares.append('0')

    elif meth == 'board':
        board_str = board.fen().split()[0]
        board_str = ''.join(board_str.split('/')[::-1])

        for x in board_str:
            if x.islower():
                squares.append('2')
            elif x.isupper():
                squares.append('1')
            elif x.isnumeric():
                squares.append('0' * int(x))
            else:
                pass
    else:
        return None

    return ''.join(squares)


def label_squares(board_img, recom_move=None, alpha=0.2):
    """
    Overlay chess coordinates onto board image
    """
    def sq_offset(sq, xr, yr):
        x = (sq % CHESSBOARD_WIDTH)
        y = CHESSBOARD_WIDTH - 1 - int(sq / CHESSBOARD_WIDTH)
        x = int((x + xr) * SQUARE_WIDTH)
        y = int((y + yr) * SQUARE_WIDTH)
        return x, y

    overlay = board_img.copy()
    for sq in range(CHESSBOARD_SQUARES):
        x, y = sq_offset(sq, 0.1, 0.8)
        cv2.putText(overlay, chess.SQUARE_NAMES[sq], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 2)

    cv2.addWeighted(overlay, alpha, board_img, 1 - alpha, 0, board_img)

    if recom_move:
        from_sq = chess.SQUARE_NAMES.index(str(recom_move)[0:2])
        to_sq = chess.SQUARE_NAMES.index(str(recom_move)[2:4])
        board_img = cv2.line(board_img, sq_offset(from_sq, 0.5, 0.5), sq_offset(to_sq, 0.5, 0.5), (0, 0, 255), 3)

        board_img = cv2.circle(board_img, sq_offset(from_sq, 0.5, 0.5), int(SQUARE_WIDTH / 2), (0, 0, 255), 5)
        board_img = cv2.circle(board_img, sq_offset(to_sq, 0.5, 0.5), int(SQUARE_WIDTH / 7), (0, 0, 255), -1)

    return board_img


def detect_move(board1, board2, meth='string', board=None):
    def input_move():
        input_move = input('Legal move not detected. Input UCI move: ')
        from_sq = chess.SQUARE_NAMES.index(input_move[0:2])
        to_sq = chess.SQUARE_NAMES.index(input_move[2:4])
        return from_sq, to_sq

    changes = 0
    changed_sq = []

    if meth == 'string': # compare using piece colour strings
        for sq in range(CHESSBOARD_SQUARES):
            if board1[sq] != board2[sq]:
                changes += 1
                changed_sq.append(sq)
                if board2[sq] == '0':
                    from_sq = sq
                if board2[sq] != '0':
                    to_sq = sq
        changed_sq = sorted(changed_sq)
        if changes == 0:
            return None
        elif changes == 2:
            return from_sq, to_sq
        # castling
        elif changes == 4:
            # naive checking
            if changed_sq == [0, 2, 3, 4]:
                return 4, 2
            elif changed_sq == [4, 5, 6, 7]:
                return 4, 6
            elif changed_sq == [56, 58, 59, 60]:
                return 60, 58
            elif changed_sq == [60, 61, 62, 63]:
                return 60, 62
            else:
                return input_move()
        else:
            return input_move()
    elif meth == 'image diff': # compare using diff between current and previous board image
        sq_dict = {
            sq:np.mean(
                cv2.cvtColor(get_square_image(board1, sq), cv2.COLOR_BGR2GRAY) -
                cv2.cvtColor(get_square_image(board2, sq), cv2.COLOR_BGR2GRAY)
            )
            for sq in range(CHESSBOARD_SQUARES)
        }
        sq_sorted = sorted(sq_dict, key=sq_dict.get, reverse=True)

        if chess.Move(sq_sorted[0], sq_sorted[1]) in board.legal_moves:
            return sq_sorted[0], sq_sorted[1]
        elif chess.Move(sq_sorted[1], sq_sorted[0]) in board.legal_moves:
            return sq_sorted[1], sq_sorted[0]
        else:
            return None


def calibrate(args):
    def get_chessboard_corners(img, draw_corners=False):
        """
        From OpenCV documentation
        """
        w = CHESSBOARD_WIDTH - 1 # OpenCV alg only detects inner corners

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # find the corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w, w), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            cnr = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            if draw_corners:
                img = cv2.drawChessboardCorners(img, (w, w), cnr, ret)

            return cnr, img
        else:
            return corners, img


    def get_board_colours(board_img, k=2):
        # Run K-Means (k=2) to get square colours
        clt = KMeans(n_clusters=2)
        clt.fit(board_img.reshape((board_img.shape[0] * board_img.shape[1], board_img.shape[2])))
        square_colours = clt.cluster_centers_.astype("uint8")

        return square_colours


    if args.calibrate == 1: # 1st calibration image - blank board
        if args.camera == -1:
            img = cv2.imread('./images/gen_calib1.jpg')
        else:
            # live corner detection with preview
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CV_CAP_PROP_FRAME_HEIGHT)
            while True:
                ret, img = cap.read()
                _, img = get_chessboard_corners(img, draw_corners=True)
                cv2.imshow('Once the coloured lines appear - press y to confirm', img)
                if cv2.waitKey(1) & 0xFF == ord('y'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            # final image capture for calibration
            img = get_single_image()
            print('Camera resolution', img.shape[0:2])

        cv2.imwrite('./calibration/empty-board.jpg', img)

        # final corner detection
        corners, calib_img1 = get_chessboard_corners(img)

        # check if valid then save
        if corners is None:
            raise Exception('No corners found')
        elif corners.shape != (49, 1, 2):
            raise Exception('Incorrect number of corners found')
        else:
            # save corners
            np.save('./calibration/corners.npy', corners)

            # save projected blank board
            board_img = project_board(calib_img1, corners)
            #square_colours = get_board_colours(board_img)
            board_img = label_squares(board_img)
            cv2.imwrite('./calibration/empty-board-projected.jpg', board_img)

            print('Calibration step 1 completed - corners detected and saved')

    elif args.calibrate == 2: # 2nd calibration image - starting board
        # load corners
        try:
            corners = np.load('./calibration/corners.npy')
        except:
            raise Exception('Calibration step 1 not run yet')

        if args.camera == -1:
            calib_img = cv2.imread('./images/gen_calib2.jpg')
        else:
            calib_img = get_single_image()

        # train piece detection model as save to file
        calib_img2 = project_board(calib_img, corners=corners)
        piece_model = fit_piece_colours(calib_img2)
        with open('./calibration/piece_model.pkl', 'wb') as f:
            pickle.dump(piece_model, f)
        print('Piece model trained and saved')

        # save second board calibration image
        cv2.imwrite('./calibration/starting-board-projected.jpg', calib_img2)

        print('Calibration step 2 completed')

    elif args.step is None:
        print('No step selected')


def message_img(board, last_move, recom_move):
    """
    Display chess metadata / stats by drawing an image. (UI of Chessbot)
    """
    # blank canvas
    img = np.zeros((400,400,1), np.uint8)

    cv2.putText(img, 'Move: {0}'.format(last_move), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(img, 'Turn: ' + chess.COLOR_NAMES[board.turn], (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    if board.is_check():
        cv2.putText(img, 'CHECK', (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    elif board.is_checkmate():
        cv2.putText(img, 'CHECKMATE', (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(img, 'Moves: {0}'.format(board.fullmove_number), (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(img, 'Board: {0}'.format(['Invalid', 'Valid'][board.is_valid()]), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    #has_castling_rights(1)
    cv2.putText(img, 'Hint: {0}'.format(recom_move), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img


def play(args):
    def recommend_move():
        """
        Recommend move using by looking up polyglot opening book. If not in book then use stockfish engine.
        """
        try:
            with polyglot.open_reader("./data/polyglot/performance.bin") as reader:
                main_entry = reader.find(board)
            recom_move = main_entry.move()
        except IndexError:
            engine = uci.popen_engine("./Stockfish/src/stockfish")
            engine.uci()
            engine.position(board)
            # Gets tuple of bestmove and ponder move.
            best_move = engine.go(movetime=ENGINE_MOVETIME)
            recom_move = best_move[0]
            engine.quit()
        return recom_move


    def display_image_window(name, img, x, y):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.resizeWindow(name, UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT)
        cv2.moveWindow(name, int(x), int(y))
        return True


    # initialise chessbaord
    board = chess.Board()

    test_img_index = 1
    prev_board_img = None
    recom_move = None

    while True:
        print(board, board.fen())
        sq_pc = get_piece_colours(None, board=board, meth='board')

        # get image
        if args.camera == -1:
            img = cv2.imread('./images/gen_{0}.jpg'.format(test_img_index))
            test_img_index += 1
        else:
            img = get_single_image()

        # crop image to board
        curr_board_img = project_board(img, corners)
        if prev_board_img is None:
            prev_board_img = curr_board_img

        # detect move

        curr_sq_pc = get_piece_colours(curr_board_img)
        curr_move = detect_move(sq_pc, curr_sq_pc)
        curr_move2 = detect_move(curr_board_img, prev_board_img, meth='image diff', board=board)
        print(sq_pc, 'prev')
        print(curr_sq_pc, 'curr')
        print(format_squares_piece_text(curr_sq_pc))
        print(curr_move, curr_move2)

        if curr_move is None:
            pass
        else:
            board_move = chess.Move(curr_move[0], curr_move[1])
            if board_move in board.legal_moves:
                board.push(board_move)
                recom_move = recommend_move()
                stat_img = message_img(board, board_move.uci(), recom_move)
            else:
                raise Exception('illegal move')

        # display UI
        display_image_window('Current Board', label_squares(curr_board_img, recom_move=recom_move), 0 * UI_WINDOW_WIDTH, 0 * UI_WINDOW_HEIGHT)
        try:
            display_image_window('Status', stat_img, 1 * UI_WINDOW_WIDTH, 0 * UI_WINDOW_HEIGHT)
        except:
            pass
        display_image_window('Raw Feed', img, 2 * UI_WINDOW_WIDTH, 0 * UI_WINDOW_HEIGHT)
        display_image_window('Diff Board', cv2.cvtColor(curr_board_img, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(prev_board_img, cv2.COLOR_BGR2GRAY), 3 * UI_WINDOW_WIDTH, 0 * UI_WINDOW_HEIGHT)

        prev_board_img = curr_board_img
        recom_move = None
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--calibrate',
        type=int,
        help="Calibration step - {1 - blank board, 2 - starting board}",
        default=None
    )
    parser.add_argument(
        '--camera',
        type=int,
        help="Camera index - {0, 1, ..., -1 for test images}",
        default=1
    )
    parser.add_argument(
        '--enginetime',
        type=int,
        help="Move time limit for engine (ms)",
        default=1000
    )
    
    args = parser.parse_args()

    CAMERA_INDEX = args.camera
    ENGINE_MOVETIME = args.enginetime

    if args.calibrate:
        calibrate(args)
    else:
        # load corners from calibration 1
        try:
            corners = np.load('./calibration/corners.npy')
            print('corners and empty board loaded')
        except:
            raise Exception('Calibration step 1 not run yet')

        # load piece detection model from calibration 2
        try:
            with open('./calibration/piece_model.pkl', 'rb') as f:
                piece_model = pickle.load(f)
            print('piece model loaded')
        except:
            raise Exception('Calibration step 2 not run yet')

        play(args)

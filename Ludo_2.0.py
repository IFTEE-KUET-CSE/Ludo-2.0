import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
import time
from random import randint, choice
import csv
import os

import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class DecisonTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, X, y, feature_index, threshold):
        parent_entropy = self.entropy(y)
        
        left_mask = X[:, feature_index] <= threshold
        left_y = y[left_mask]
        left_entropy = self.entropy(left_y)
        left_weight = len(left_y) / len(y)

        right_mask = X[:, feature_index] > threshold
        right_y = y[right_mask]
        right_entropy = self.entropy(right_y)
        right_weight = len(right_y) / len(y)

        information_gain = parent_entropy - (left_weight * left_entropy) - (right_weight * right_entropy)
        return information_gain

    def best_split(self, X, y):
        best_gain = 0
        best_feature_index = None
        best_threshold = None
        n_features = X.shape[1]

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return DecisionNode(value=y[0])

        if self.max_depth is not None and depth >= self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            majority_class = values[np.argmax(counts)]
            return DecisionNode(value=majority_class)

        feature_index, threshold = self.best_split(X, y)
        if feature_index is None or threshold is None:
            values, counts = np.unique(y, return_counts=True)
            majority_class = values[np.argmax(counts)]
            return DecisionNode(value=majority_class)

        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        left_node = self.build_tree(X[left_mask], y[left_mask], depth+1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth+1)

        return DecisionNode(feature_index=feature_index, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self.predict_sample(x, self.tree)
            predictions.append(prediction)
        return np.array(predictions)


class Ludo:
    def __init__(self, root,six_side_block,five_side_block,four_side_block,three_side_block,two_side_block,one_side_block):
        self.classifier = self.model()
        self.window = root
        
        # Make canvas
        self.make_canvas = Canvas(self.window, bg="#141414", width=800, height=630)
        self.make_canvas.pack(fill=BOTH,expand=1)
        self.output_li = []
        
        self.made_red_coin = []     # ui coin
        self.made_sky_blue_coin = []

        self.red_number_label = []
        self.sky_blue_number_label = []
        self.choice_dice = False

        self.block_value_predict = []   
        self.total_people_play = []

        self.odd_even_human = 3
        self.odd_even_computer = 3

        self.double_dice = 3

        # dice block all side image store
        self.block_number_side = [one_side_block, two_side_block, three_side_block, four_side_block, five_side_block, six_side_block]

        # store specific position of all coins
        self.red_coord_store = [-1, -1, -1, -1]
        self.sky_blue_coord_store = [-1, -1, -1, -1]

        self.red_coin_position = [-1, -1, -1, -1]
        self.sky_blue_coin_position = [-1, -1, -1, -1]

        # Number to room to be traverse by specific color coin, store in that variable
        self.move_red_counter = 0
        self.move_sky_blue_counter = 0

        self.take_permission = 0   # is finished check
        self.six_with_overlap = 0

        self.six_counter = 0
        self.time_for = -1  # which player will role

        # Robo Control
        self.robo_prem = 0
        self.count_robo_stage_from_start = 0
        self.robo_store = []

        self.current_coin_moved = 0

        self.is_sky_blue_odd = False
        self.is_sky_blue_even = False
        self.is_red_odd = False
        self.is_red_even = False

        self.is_double_dice = False
        
        self.board_set_up()
        self.init_btn_red()
        self.init_btn_sky_blue()
        self.take_initial_control()


    def board_set_up(self):
        # Cover Box made
        self.make_canvas.create_rectangle(100, 15, 100 + (40 * 15), 15 + (40 * 15), width=6, fill="white")

        # Square box
        self.make_canvas.create_rectangle(100, 15, 100+240, 15+240, width=3, fill="#C96731")# left up large square
        self.make_canvas.create_rectangle(100, (15+240)+(40*3), 100+240, (15+240)+(40*3)+(40*6), width=3, fill="#F6A21D")# left down large square
        self.make_canvas.create_rectangle(340+(40*3), 15, 340+(40*3)+(40*6), 15+240, width=3, fill="#F6A21D")# right up large square
        self.make_canvas.create_rectangle(340+(40*3), (15+240)+(40*3), 340+(40*3)+(40*6), (15+240)+(40*3)+(40*6), width=3, fill="#C96731")# right down large square

        # Left 3 box(In white region)
        self.make_canvas.create_rectangle(100, (15+240), 100+240, (15+240)+40, width=3)
        self.make_canvas.create_rectangle(100+40, (15 + 240)+40, 100 + 240, (15 + 240) + 40+40, width=3, fill="#C96731")
        self.make_canvas.create_rectangle(100, (15 + 240)+80, 100 + 240, (15 + 240) + 80+40, width=3)

        # right 3 box(In white region)
        self.make_canvas.create_rectangle(100+240, 15, 100 + 240+40, 15 + (40*6), width=3)
        self.make_canvas.create_rectangle(100+240+40, 15+40, 100+240+80, 15 + (40*6), width=3, fill="#F6A21D")
        self.make_canvas.create_rectangle(100+240+80, 15, 100 + 240+80+40, 15 + (40*6), width=3)

        # up 3 box(In white region)
        self.make_canvas.create_rectangle(340+(40*3), 15+240, 340+(40*3)+(40*6), 15+240+40, width=3)
        self.make_canvas.create_rectangle(340+(40*3), 15+240+40, 340+(40*3)+(40*6)-40, 15+240+80, width=3, fill="#C96731")
        self.make_canvas.create_rectangle(340+(40*3), 15+240+80, 340+(40*3)+(40*6), 15+240+120, width=3)

        # down 3 box(In white region)
        self.make_canvas.create_rectangle(100, (15 + 240)+(40*3), 100 + 240+40, (15 + 240)+(40*3)+(40*6), width=3)
        self.make_canvas.create_rectangle(100+240+40, (15 + 240)+(40*3), 100 + 240+40+40, (15 + 240)+(40*3)+(40*6)-40, width=3, fill="#F6A21D")
        self.make_canvas.create_rectangle(100 + 240+40+40, (15 + 240)+(40*3), 100 + 240+40+40+40, (15 + 240)+(40*3)+(40*6), width=3)

        # All left separation line
        start_x = 100 + 40
        start_y = 15 + 240
        end_x = 100 + 40
        end_y = 15 + 240 + (40 * 3)
        for _ in range(5):
            self.make_canvas.create_line(start_x, start_y, end_x, end_y, width=3)
            start_x+=40
            end_x+= 40

        # All right separation line
        start_x = 100+240+(40*3)+40
        start_y = 15 + 240
        end_x = 100+240+(40*3)+40
        end_y = 15 + 240 + (40 * 3)
        for _ in range(5):
            self.make_canvas.create_line(start_x, start_y, end_x, end_y, width=3)
            start_x += 40
            end_x += 40

        # All up separation done
        start_x = 100+240
        start_y = 15+40
        end_x = 100+240+(40*3)
        end_y = 15+40
        for _ in range(5):
            self.make_canvas.create_line(start_x, start_y, end_x, end_y, width=3)
            start_y += 40
            end_y += 40

        # All down separation done
        start_x = 100 + 240
        start_y = 15 + (40*6)+(40*3)+40
        end_x = 100 + 240 + (40 * 3)
        end_y = 15 + (40*6)+(40*3)+40
        for _ in range(5):
            self.make_canvas.create_line(start_x, start_y, end_x, end_y, width=3)
            start_y += 40
            end_y += 40

        # sky_blue start position
        self.make_canvas.create_rectangle(100+240,340+(40*5)-5,100+240+40,340+(40*6)-5,fill="#F6A21D",width=3)
        # Red start position
        self.make_canvas.create_rectangle(100 + 40, 15+(40*6), 100 +40 + 40, 15+(40*6)+40, fill="#C96731", width=3)
        # Green start position
        self.make_canvas.create_rectangle(100 + (40*8), 15 + 40, 100 +(40*9), 15 + 40+ 40, fill="#F6A21D", width=3)
        # Yellow start position
        self.make_canvas.create_rectangle(100 + (40 * 6)+(40*3)+(40*4), 15 + (40*8), 100 + (40 * 6)+(40*3)+(40*5), 15 + (40*9), fill="#C96731", width=3)

        # square in middle
        self.make_canvas.create_rectangle(100+240, 15+40+(40*5), 340+(40*4)+40+60+40+20-200, 340+80+60+40+15-(40*4), width=3, fill="#9CA383")

        # Make coin for red left up block
        red_1_coin = self.make_canvas.create_oval(100+40, 15+40, 100+40+40, 15+40+40, width=3, fill="red", outline="black")
        red_2_coin = self.make_canvas.create_oval(100+40+60+60, 15 + 40, 100+40+60+60+40, 15 + 40 + 40, width=3, fill="red", outline="black")
        red_3_coin = self.make_canvas.create_oval(340 + (40 * 3) + 40 + 60 + 40 + 20, 340 + 80 + 60 + 40 + 15, 340 + (40 * 3) + 40 + 60 + 40 + 40 + 20, 340 + 80 + 60 + 40 + 40 + 15, width=3, fill="red", outline="black")
        red_4_coin = self.make_canvas.create_oval(340 + (40 * 3) + 40, 340+80+60+40+15, 340 + (40 * 3) + 40 + 40,340+80+60+40+40+15, width=3, fill="red", outline="black")
        self.made_red_coin.append(red_1_coin)
        self.made_red_coin.append(red_2_coin)
        self.made_red_coin.append(red_3_coin)
        self.made_red_coin.append(red_4_coin)

        # dice choice coordinate
        self.make_canvas.create_oval(100+240, 15+40+(40*4), 100+240+40, 15+40+40+(40*4), width=3, fill="#000000", outline="black")
        red_dice_choice = Label(self.make_canvas, text="choice", font=("Arial", 7, "bold"), bg="#000000", fg="white")
        red_dice_choice.place(x=100+240+2, y=15+40+11+(40*4))

        self.make_canvas.create_oval(340+(40*3)+40+60+40+20-200, 340+80+60+40+15-(40*4), 340+(40*3)+40+60+40+40+20-200, 340+80+60+40+40+15-(40*4), width=3, fill="#000000", outline="black")
        yellow_dice_choice = Label(self.make_canvas, text="choice", font=("Arial", 7, "bold"), bg="#000000", fg="white")
        yellow_dice_choice.place(x=340+(40*3)+40+60+40+20-200+2, y=340+80+60+40+15+11-(40*4))

        self.make_canvas.create_oval(340+(40*3)+40+60+40+20-160, 15+40+(40*5), 340+(40*3)+40+60+40+40+20-160, 15+40+40+(40*5), width=3, fill="#000000", outline="black")
        green_dice_choice = Label(self.make_canvas, text="choice", font=("Arial", 7, "bold"), bg="#000000", fg="white")
        green_dice_choice.place(x=340+(40*3)+40+60+40+20-160+2, y=15+40+11+(40*5))

        self.make_canvas.create_oval(100+200, 340+80+60+40+15-(40*5), 100 + 200 + 40, 340+80+60+40+40+15-(40*5), width=3, fill="#000000", outline="black")
        sky_blue_dice_choice = Label(self.make_canvas, text="choice", font=("Arial", 7, "bold"), bg="#000000", fg="white")
        sky_blue_dice_choice.place(x=100+200+2, y=340+80+60+40+15+11-(40*5))

        # Make coin under number label for red left up block
        red_1_label = Label(self.make_canvas, text="1", font=("Arial", 15, "bold"), bg="red", fg="black")
        red_1_label.place(x=100 + 40 + 10, y=15 + 40 + 5)
        red_2_label = Label(self.make_canvas, text="2", font=("Arial", 15, "bold"), bg="red", fg="black")
        red_2_label.place(x=100 + 40 + 60 + 60 + 10, y=15 + 40 + 5)
        red_3_label = Label(self.make_canvas, text="3", font=("Arial", 15, "bold"), bg="red", fg="black")
        red_3_label.place(x=340 + (40 * 3) + 40 + 40 + 60 + 30, y=30 + (40 * 6) + (40 * 3) + 40 + 100 + 10)
        red_4_label = Label(self.make_canvas, text="4", font=("Arial", 15, "bold"), bg="red", fg="black")
        red_4_label.place(x=340 + (40 * 3) + 40 + 10, y=30 + (40 * 6) + (40 * 3) + 40 + 100 + 10)
        self.red_number_label.append(red_1_label)
        self.red_number_label.append(red_2_label)
        self.red_number_label.append(red_3_label)
        self.red_number_label.append(red_4_label)

        # Make coin for sky_blue left down block
        sky_blue_1_coin = self.make_canvas.create_oval(340+(40*3)+40, 15 + 40, 340+(40*3)+40 + 40, 15 + 40 + 40, width=3, fill="#04d9ff", outline="black")
        sky_blue_2_coin = self.make_canvas.create_oval(340+(40*3)+40+ 60 + 40+20, 15 + 40, 340+(40*3)+40 + 60 + 40 + 40+20, 15 + 40 + 40, width=3, fill="#04d9ff", outline="black")
        sky_blue_3_coin = self.make_canvas.create_oval(100 + 40 + 60 + 40 + 20, 340 + 80 + 60 + 40 + 15, 100 + 40 + 60 + 40 + 40 + 20, 340 + 80 + 60 + 40 + 40 + 15, width=3, fill="#04d9ff", outline="black")
        sky_blue_4_coin = self.make_canvas.create_oval( 100 + 40, 340+80+60+40+15, 100 + 40 + 40, 340+80+60+40+40+15, width=3, fill="#04d9ff", outline="black")
        self.made_sky_blue_coin.append(sky_blue_1_coin)
        self.made_sky_blue_coin.append(sky_blue_2_coin)
        self.made_sky_blue_coin.append(sky_blue_3_coin)
        self.made_sky_blue_coin.append(sky_blue_4_coin)

        # Make coin under number label for sky_blue left down block
        sky_blue_1_label = Label(self.make_canvas, text="1", font=("Arial", 15, "bold"), bg="#04d9ff", fg="black")
        sky_blue_1_label.place(x=340 + (40 * 3) + 40 + 10, y=15 + 40 + 5)
        sky_blue_2_label = Label(self.make_canvas, text="2", font=("Arial", 15, "bold"), bg="#04d9ff", fg="black")
        sky_blue_2_label.place(x=340 + (40 * 3) + 40 + 40 + 60 + 30, y=15 + 40 + 5)
        sky_blue_3_label = Label(self.make_canvas, text="3", font=("Arial", 15, "bold"), bg="#04d9ff", fg="black")
        sky_blue_3_label.place(x=100 + 40 + 60 + 60 + 10, y=30 + (40 * 6) + (40 * 3) + 40 + 60 + 40 + 10)
        sky_blue_4_label = Label(self.make_canvas, text="4", font=("Arial", 15, "bold"), bg="#04d9ff", fg="black")
        sky_blue_4_label.place(x=100 + 40 + 10, y=30 + (40 * 6) + (40 * 3) + 40 + 60 + 40 + 10)
        self.sky_blue_number_label.append(sky_blue_1_label)
        self.sky_blue_number_label.append(sky_blue_2_label)
        self.sky_blue_number_label.append(sky_blue_3_label)
        self.sky_blue_number_label.append(sky_blue_4_label)

        btn_odd_sky_blue = Button(self.make_canvas, bg="black", fg="#00FF00", relief=RAISED, bd=5, text="Odd",font=("Arial", 8, "bold"), command=lambda: self.odd_sky_blue())
        btn_odd_sky_blue.place(x=5, y=(40*6+40*2)+20)
        btn_even_sky_blue = Button(self.make_canvas, bg="black", fg="#00FF00", relief=RAISED, bd=5, text="Even",font=("Arial", 8, "bold"), command=lambda: self.even_sky_blue())
        btn_even_sky_blue.place(x=55, y=(40*6+40*2)+20)

        btn_odd_sky_blue = Button(self.make_canvas, bg="black", fg="#00FF00", relief=RAISED, bd=5, text="Double dice",font=("Arial", 8, "bold"), command=lambda: self.double_dice_fun())
        btn_odd_sky_blue.place(x=12, y=(40*6+40+20))

        # Make star safe zone
        self.make_canvas.create_oval(100+240, 15+40+(40*1), 100+240+40, 15+40+40+(40*1), width=3, fill="blue", outline="black")
        red_star_zone = Label(self.make_canvas, text="star", font=("Arial", 10, "bold"), bg="blue", fg="white")
        red_star_zone.place(x=100+240+6, y=15+40+8+(40*1))

        self.make_canvas.create_oval(340+(40*3)+40+60+40+20-200, 340+80+60+40+15-(40*1), 340+(40*3)+40+60+40+40+20-200, 340+80+60+40+40+15-(40*1), width=3, fill="blue", outline="black")
        yellow_star_zone = Label(self.make_canvas, text="star", font=("Arial", 10, "bold"), bg="blue", fg="white")
        yellow_star_zone.place(x=340+(40*3)+40+60+40+20-200+6, y=340+80+60+40+15+8-(40*1))

        self.make_canvas.create_oval(340+(40*6)+40+60+40+20-160, 15+40+(40*5), 340+(40*6)+40+60+40+40+20-160, 15+40+40+(40*5), width=3, fill="blue", outline="black")
        green_star_zone = Label(self.make_canvas, text="star", font=("Arial", 10, "bold"), bg="blue", fg="white")
        green_star_zone.place(x=340+(40*6)+40+60+40+20-160+6, y=15+40+8+(40*5))

        self.make_canvas.create_oval(100+200-(40*3), 340+80+60+40+15-(40*5), 100 + 200 + 40 - (40*3), 340+80+60+40+40+15-(40*5), width=3, fill="blue", outline="black")
        sky_blue_star_zone = Label(self.make_canvas, text="star", font=("Arial", 10, "bold"), bg="blue", fg="white")
        sky_blue_star_zone.place(x=100+200+6-(40*3), y=340+80+60+40+15+8-(40*5))

    def odd_sky_blue(self):
        print("odd sky_blue")
        if self.odd_even_human != 0:
            self.is_sky_blue_odd = True
            self.odd_even_human -= 1
            print("odd_even power remaining ", self.odd_even_human)
            self.make_prediction("sky_blue")
        else:
            print("odd_even power finish")

    def even_sky_blue(self):
        print("even sky_blue")
        if self.odd_even_human != 0:
            self.is_sky_blue_even = True
            self.odd_even_human -= 1
            print("odd_even power remaining ", self.odd_even_human)
            self.make_prediction("sky_blue")
        else:
            print("odd_even power finish")

    def double_dice_fun(self):
        print("double dice")
        if self.double_dice != 0:
            self.is_double_dice = True
            self.double_dice -= 1
            print("Double dice power remaining ", self.double_dice)
            self.make_prediction("sky_blue")
        else:
            print("double dice power finished")

    # Control take at first
    def take_initial_control(self):
        for i in range(2):
            self.block_value_predict[i][1]['state'] = DISABLED

        self.robo_prem = 1
        for player_index in range(2):
            self.total_people_play.append(player_index)
        print(self.total_people_play)
        self.block_value_predict[1][1]['state'] = NORMAL


    # Define membership functions
    def membership_risk_of_capture(self,distance):
        degree1 ={}
        if distance >= 1 and distance <= 3:
            degree1['low'] = 0
            degree1['medium'] = 0
            degree1['high'] = 1
        elif distance > 3 and distance <=6:
            degree1['low'] = 0
            degree1['medium'] = float((6-distance)*(1.0/(6-3)))
            degree1['high'] =  float((distance-3)*(1.0/(6-3)))
        elif distance > 6 and distance <= 8:
            degree1['low'] = 0
            degree1['medium'] = 1
            degree1['high'] = 0
        elif distance > 8 and distance <= 12:
            degree1['low'] = float((12-distance)*(1.0/(12-8)))
            degree1['medium'] = float((distance-8)*(1.0/(12-8)))
            degree1['high'] = 0
        else:
            degree1['low'] = 1
            degree1['medium'] = 0
            degree1['high'] = 0
        return degree1

    def membership_try_to_capture(self,distance):
        degree2 ={}
        if (distance >= 1 and distance <= 3 ):
            degree2['low'] = 0
            degree2['medium'] = 0
            degree2['high'] = 1
        elif distance > 3 and distance <=6:
            degree2['low'] = 0
            degree2['medium'] = float((distance-3)*(1.0/(6-3)))
            degree2['high'] = float((6-distance)*(1.0/(6-3)))
        elif distance > 6 and distance <= 8:
            degree2['low'] = 0
            degree2['medium'] = 1
            degree2['high'] = 0
        elif distance > 8 and distance <= 12:
            degree2['low'] = float((12-distance)*(1.0/(12-8)))
            degree2['medium'] = float((distance-8)*(1.0/(12-8)))
            degree2['high'] = 0
        else:
            degree2['low'] = 1
            degree2['medium'] = 0
            degree2['high'] = 0
        return degree2

    def membership_isRotten(self, life_distance):
        degree3 ={}
        if life_distance >= 1 and life_distance <= 20:
            degree3['low'] = 1
            degree3['high'] = 0
        elif life_distance > 20 and life_distance <= 40:
            degree3['low'] = float((40-life_distance)*(1.0/(40-20)))
            degree3['high'] = float((life_distance-20)*(1.0/(40-20)))
        else:
            degree3['low'] = 0
            degree3['high'] = 1
        return degree3

    def ruleEvalationAssessment(self,backward_distance, forward_distance,life_distance):
        to_be_captured=self.membership_risk_of_capture(backward_distance)
        try_to_capture=self.membership_try_to_capture(forward_distance)
        rotten_value = self.membership_isRotten(life_distance)
        dice_low= min(try_to_capture['high'],rotten_value['low'])
        dice_star= min(to_be_captured['high'],rotten_value['high'])
        dice_high= max(to_be_captured['low'],max(try_to_capture['medium'],try_to_capture['low'],rotten_value['low']))
        
        return dice_low,dice_star,dice_high
    
    def defuzzificationAssessment(self, backward_distance, forward_distance, life_distance):
        
        dice_low,dice_star,dice_high = self.ruleEvalationAssessment(backward_distance, forward_distance,life_distance)
        up=0
        down=0
        x=0
        while x<=100:
            
            if(x>=0 and x<=50):
                up= up+x*dice_low
                down =down+dice_low
            elif (x>50 and x<=80):
                up= up+x*dice_star
                down =down+dice_star
            elif (x>80 and x<=100):
                up= up+x*dice_high
                down =down+dice_high
            x=x+1  
        return up/down



    # Get block value after prediction based on probability
    def make_prediction(self,color_indicator):  #kon gutir khela akhon, dice roll a koto value uthce, r block value precict (boro list) ta set kora.
            
            if self.choice_dice == True:       
                if color_indicator == "red":
                    i = self.current_coin_moved - 1
                    dis_difference = 100000000000
                    for j in range(4):
                        if self.red_coin_position[i] == -1 or self.sky_blue_coin_position[i] >= 100:
                            dis_difference = -100
                            break

                        if self.sky_blue_coin_position[j] == -1 or self.sky_blue_coin_position[j] >= 100:
                            dis_difference = 100
                            continue


                        if self.sky_blue_coin_position[j] < self.red_coin_position[i]:
                            dis_difference = min(dis_difference,self.red_coin_position[i] - self.sky_blue_coin_position[j])
                        else:
                            dis_difference = min(dis_difference, 52 - self.sky_blue_coin_position[j] + self.red_coin_position[i])
                    backward_distance = dis_difference
                    
                    capture_dis_difference = 100000000000
                    for j in range(4):
                            if self.red_coin_position[i] == -1 or self.red_coin_position[i] >= 100:
                                capture_dis_difference = -100
                                break
                            if self.sky_blue_coin_position[j] == -1 or self.sky_blue_coin_position[j] >= 100:
                                capture_dis_difference = 100
                                continue
                            if self.sky_blue_coin_position[j] > self.red_coin_position[i]:
                                capture_dis_difference = min(capture_dis_difference,self.sky_blue_coin_position[i] - self.red_coin_position[j])
                            else:
                                capture_dis_difference = min(capture_dis_difference, 52 - self.red_coin_position[i] + self.sky_blue_coin_position[j])
                    forward_distance = capture_dis_difference
                    if i==0 or i==1:
                        life_distance = self.red_coin_position[i];
                    elif i==2 or i==3:
                        if self.red_coin_pos >= 27 and self.red_coin_position[i] <= 52:
                            life_distance = 52 - self.red_coin_position[i]
                        else:
                            life_distance = (52-27)+self.red_coin_position[i]

                    
                    conAssessment=self.defuzzificationAssessment(backward_distance, forward_distance, life_distance)
                    if conAssessment>=0 and conAssessment<=50:
                        p_n=-1
                        for y in range(3):
                            if self.red_coord_store[i] + y in self.sky_blue_coord_store:
                                p_n=y
                                break
                        if p_n==-1:
                            p_n=4
                        x=p_n
                    elif conAssessment>50 and conAssessment<=80:
                        x=4
                    elif conAssessment>80 and conAssessment<=100:
                        x=5
                            
                        
                    block_value_predict = self.block_value_predict[0]
                    if self.robo_prem and self.count_robo_stage_from_start < 3:
                        self.count_robo_stage_from_start += 1
                    if self.robo_prem and self.count_robo_stage_from_start == 3 and self.six_counter < 2:
                        permanent_block_number = self.move_red_counter = x
                        self.count_robo_stage_from_start += 1
                    else:    
                        permanent_block_number = self.move_red_counter = x

                elif color_indicator == "sky_blue":
                    y = randint(1, 5)
                    block_value_predict = self.block_value_predict[1]
                    permanent_block_number = self.move_sky_blue_counter = y
                    if self.robo_prem and permanent_block_number == 6:             
                        for coin_loc in self.red_coin_position:
                            if coin_loc>=40 and coin_loc<=46:
                                permanent_block_number = self.move_sky_blue_counter = y
                                break
                
                self.time_for -= 1
                self.choice_dice = False

            elif color_indicator == "red":
                x = randint(1, 6)
                block_value_predict = self.block_value_predict[0]
                if self.robo_prem and self.count_robo_stage_from_start < 3:
                    self.count_robo_stage_from_start += 1
                if self.robo_prem and self.count_robo_stage_from_start == 3 and self.six_counter < 2:
                    permanent_block_number = self.move_red_counter = x
                    self.count_robo_stage_from_start += 1
                else:    
                    permanent_block_number = self.move_red_counter = x

            elif color_indicator == "sky_blue":
                y = randint(1, 6)
                block_value_predict = self.block_value_predict[1]
                permanent_block_number = self.move_sky_blue_counter = y
                if self.robo_prem and permanent_block_number == 6:              
                    for coin_loc in self.red_coin_position:
                        if coin_loc>=40 and coin_loc<=46:
                            permanent_block_number = self.move_sky_blue_counter = y
                            break
        
            block_value_predict[1]['state'] = DISABLED

            odd = [1, 3, 5]
            even = [2, 4, 6]
            if self.is_sky_blue_odd == True:
                permanent_block_number = self.move_sky_blue_counter = odd[randint(0, 2)]
                self.is_sky_blue_odd = False

            if self.is_sky_blue_even == True:
                permanent_block_number = self.move_sky_blue_counter = even[randint(0, 2)]
                self.is_sky_blue_even = False

            if self.is_red_odd == True:
                permanent_block_number = self.move_red_counter = odd[randint(0, 2)]
                self.is_red_odd = False

            if self.is_red_even == True:
                permanent_block_number = self.move_red_counter = even[randint(0, 2)]
                self.is_red_even = False

            if self.is_double_dice == True and color_indicator == "sky_blue":
                self.move_sky_blue_counter = 2*permanent_block_number
                self.is_double_dice = False

            # Illusion of coin
            temp_counter = 12
            while temp_counter>0:
                move_temp_counter = randint(1, 6)
                block_value_predict[0]['image'] = self.block_number_side[move_temp_counter - 1]
                self.window.update()
                time.sleep(0.1)
                temp_counter-=1

            print("Prediction result: ", permanent_block_number)

            # Permanent predicted value containing image set
            block_value_predict[0]['image'] = self.block_number_side[permanent_block_number-1]
            if self.robo_prem == 1 and color_indicator == "red":
                self.window.update()
                time.sleep(0.4)
            self.is_possible_to_move_based_on_current_situation(color_indicator,permanent_block_number,block_value_predict)
        
    def is_possible_to_move_based_on_current_situation(self,color_indicator,permanent_block_number,block_value_predict): 
        # block value predict r color indicator r permanent block number anujayi: kokhon tmi dan dite parbe, r dan dite na parle permission =0 dibo ba porer joner kace dan pass kore dibo, kon kon guti k move korar premission dibo 
        robo_operator = None
        if color_indicator == "red":
            temp_coin_position = self.red_coin_position
        else:
            temp_coin_position = self.sky_blue_coin_position


        
        all_in = 1
        for i in range(4):
            if temp_coin_position[i] == -1:
                all_in = 1
            else:
                all_in = 0
                break

        if  permanent_block_number == 6:
            self.six_counter += 1
        else:
            self.six_counter = 0

        if ((all_in == 1 and permanent_block_number == 6) or (all_in==0)) and self.six_counter<3:
            permission = 1
            if color_indicator == "red":
                temp = self.red_coord_store
            else:
                temp = self.sky_blue_coord_store

            if  permanent_block_number<6:
                if self.six_with_overlap == 1:
                    self.time_for-=1
                    self.six_with_overlap=0
                for i in range(4):
                    if  temp[i] == -1:
                        permission=0
                    elif temp[i]>100:
                        if  temp[i]+permanent_block_number<=106:
                            permission=1
                            break
                        else:
                            permission=0
                    else:
                        permission=1
                        break
            else:
                for i in range(4):
                    if  temp[i]>100:
                        if  temp[i] + permanent_block_number <= 106:
                            permission = 1
                            break
                        else:
                            permission = 0
                    else:
                        permission = 1
                        break

            if permission == 0:
                self.make_command(None) #simply pass to next player
            else:
                
                if self.sky_blue_coord_store[self.current_coin_moved-1] == 6 or  self.sky_blue_coord_store[self.current_coin_moved-1] == 19 or self.sky_blue_coord_store[self.current_coin_moved-1] == 32 or self.sky_blue_coord_store[self.current_coin_moved-1] == 45:   # hello
                    self.main_controller("sky_blue", self.current_coin_moved)
                else:
                    self.num_btns_state_controller(block_value_predict[2])
                    block_value_predict[1]['state'] = DISABLED

                if self.robo_prem == 1 and block_value_predict == self.block_value_predict[0]:
                    robo_operator = "give"
                    block_value_predict[1]['state'] = DISABLED
                
        else:
            block_value_predict[1]['state'] = NORMAL    # Predict btn activation
            if self.six_with_overlap == 1:
                self.time_for -= 1
                self.six_with_overlap = 0
            self.make_command()

        if  permanent_block_number == 6 and self.six_counter<3 and block_value_predict[2][0]['state'] == NORMAL:
            self.time_for-=1
        else:
            self.six_counter=0

        if self.robo_prem == 1 and robo_operator:
            self.robo_judge(robo_operator)

    # Player Scope controller
    def make_command(self, robo_operator=None):
        if  self.time_for == -1:
            pass
        else:
            self.block_value_predict[self.total_people_play[self.time_for]][1]['state'] = DISABLED
        if  self.time_for == len(self.total_people_play)-1:
            self.time_for = -1

        self.time_for+=1
        self.block_value_predict[self.total_people_play[self.time_for]][1]['state'] = NORMAL
        
        if self.robo_prem==1 and self.time_for == 0:
            robo_operator = "predict"
        if robo_operator:
            self.robo_judge(robo_operator)


    def init_btn_red(self):
        block_predict_red = Label(self.make_canvas,image=self.block_number_side[0])
        block_predict_red.place(x=34,y=15)
        predict_red = Button(self.make_canvas, bg="black", fg="#00FF00", relief=RAISED, bd=5, text="Predict", font=("Arial", 8, "bold"), command=lambda: self.make_prediction("red"))
        
        btn_1 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="1",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("red",'1'), state=DISABLED, disabledforeground="red")
        btn_2 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="2",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("red",'2'), state=DISABLED, disabledforeground="red")
        btn_3 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="3",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("red",'3'), state=DISABLED, disabledforeground="red")
        btn_4 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="4",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("red",'4'), state=DISABLED, disabledforeground="red")
        
        Label(self.make_canvas,text="Comp",bg="#141414",fg="gold",font=("Arial",15,"bold")).place(x=25, y=15 + 50)
        self.store_instructional_btn(block_predict_red,predict_red,[btn_1,btn_2,btn_3,btn_4])

    def init_btn_sky_blue(self):
        block_predict_sky_blue = Label(self.make_canvas, image=self.block_number_side[0])
        block_predict_sky_blue.place(x=34, y=15+(40*6+40*3)+10)
        predict_sky_blue = Button(self.make_canvas, bg="black", fg="#00FF00", relief=RAISED, bd=5, text="Predict",font=("Arial", 8, "bold"), command=lambda: self.make_prediction("sky_blue"))
        predict_sky_blue.place(x=25, y=15+(40*6+40*3)+40 + 20)

        btn_1 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="1",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("sky_blue",'1'), state=DISABLED, disabledforeground="red")
        btn_1.place(x=20,y=15+(40*6+40*3)+40 + 70)
        btn_2 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="2",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("sky_blue",'2'), state=DISABLED, disabledforeground="red")
        btn_2.place(x=60,y=15+(40*6+40*3)+40 + 70)
        btn_3 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="3",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("sky_blue",'3'), state=DISABLED, disabledforeground="red")
        btn_3.place(x=20,y=15+(40*6+40*3)+40 + 70+ 40)
        btn_4 = Button(self.make_canvas,bg="#262626",fg="#00eb00",text="4",font=("Arial",13,"bold","italic"),relief=RAISED,bd=3,command=lambda: self.main_controller("sky_blue",'4'), state=DISABLED, disabledforeground="red")
        btn_4.place(x=60,y=15+(40*6+40*3)+40 + 70+ 40)

        Label(self.make_canvas, text="Human", bg="#141414", fg="gold", font=("Arial", 15, "bold")).place(x=12,y=15+(40*6+40*3)+40 + 110+50)
        self.store_instructional_btn(block_predict_sky_blue, predict_sky_blue, [btn_1,btn_2,btn_3,btn_4])

    def store_instructional_btn(self, block_indicator, predictor, entry_controller):
        temp = []
        temp.append(block_indicator)
        temp.append(predictor)
        temp.append(entry_controller)
        self.block_value_predict.append(temp)

    def red_circle_start_position(self, coin_number):
        self.make_canvas.delete(self.made_red_coin[int(coin_number)-1])
        self.made_red_coin[int(coin_number)-1] = self.make_canvas.create_oval(100 + 40, 15+(40*6), 100 +40 + 40, 15+(40*6)+40, fill="red", width=3, outline="black")

        self.red_number_label[int(coin_number)-1].place_forget()
        red_start_label_x = 100 + 40 + 10
        red_start_label_y = 15 + (40 * 6) + 5
        self.red_number_label[int(coin_number)-1].place(x=red_start_label_x, y=red_start_label_y)

        self.red_coin_position[int(coin_number)-1] = 1
        self.window.update()
        time.sleep(0.2)

    def green_circle_start_position(self,coin_number):
        self.make_canvas.delete(self.made_sky_blue_coin[int(coin_number)-1])
        self.made_sky_blue_coin[int(coin_number)-1] = self.make_canvas.create_oval(100 + (40*8), 15 + 40, 100 +(40*9), 15 + 40+ 40, fill="#04d9ff", width=3)

        self.sky_blue_number_label[int(coin_number)-1].place_forget()
        sky_blue_start_label_x = 100 + (40*8) + 10
        sky_blue_start_label_y = 15 + 40 + 5
        self.sky_blue_number_label[int(coin_number)-1].place(x=sky_blue_start_label_x, y=sky_blue_start_label_y)

        self.sky_blue_coin_position[int(coin_number)-1] = 14
        self.window.update()
        time.sleep(0.2)

    def yellow_circle_start_position(self,coin_number):
        self.make_canvas.delete(self.made_red_coin[int(coin_number)-1])
        self.made_red_coin[int(coin_number)-1] = self.make_canvas.create_oval(100 + (40 * 6)+(40*3)+(40*4), 15 + (40*8), 100 + (40 * 6)+(40*3)+(40*5), 15 + (40*9), fill="red", width=3)

        self.red_number_label[int(coin_number)-1].place_forget()
        red_start_label_x = 100 + (40 * 6)+(40*3)+(40*4) + 10
        red_start_label_y = 15 + (40*8) + 5
        self.red_number_label[int(coin_number) - 1].place(x=red_start_label_x, y=red_start_label_y)

        self.red_coin_position[int(coin_number) - 1] = 27
        self.window.update()
        time.sleep(0.2)

    def sky_blue_circle_start_position(self,coin_number):
        self.make_canvas.delete(self.made_sky_blue_coin[int(coin_number)-1])
        self.made_sky_blue_coin[int(coin_number)-1] = self.make_canvas.create_oval(100+240,340+(40*5)-5,100+240+40,340+(40*6)-5,fill="#04d9ff",width=3)

        self.sky_blue_number_label[int(coin_number)-1].place_forget()
        sky_blue_start_label_x = 100+240 + 10
        sky_blue_start_label_y = 340+(40*5)-5 + 5
        self.sky_blue_number_label[int(coin_number) - 1].place(x=sky_blue_start_label_x, y=sky_blue_start_label_y)
        self.sky_blue_coin_position[int(coin_number) - 1] = 40
        self.window.update()
        time.sleep(0.2)

    def num_btns_state_controller(self, take_nums_btns_list, state_control = 1):
        if state_control:
            for num_btn in take_nums_btns_list:
                num_btn['state'] = NORMAL
        else:
            for num_btn in take_nums_btns_list:
                num_btn['state'] = DISABLED

    def main_controller(self, color_coin, coin_number):
        robo_operator = None
        
        if  color_coin == "red":
            self.num_btns_state_controller(self.block_value_predict[0][2], 0)

            if self.move_red_counter == 106:
                messagebox.showwarning("Destination reached","Reached at the destination")

            elif self.red_coin_position[int(coin_number)-1] == -1 and self.move_red_counter == 6:
                if int(coin_number) == 1 or int(coin_number) == 2:
                    self.red_circle_start_position(coin_number)
                    self.red_coord_store[int(coin_number) - 1] = 1     # computer 1,2 start position
                else:
                    self.yellow_circle_start_position(coin_number)
                    self.red_coord_store[int(coin_number) - 1] = 27    # computer 3,4 start position

            elif self.red_coin_position[int(coin_number)-1] > -1:
                take_coord = self.make_canvas.coords(self.made_red_coin[int(coin_number)-1])
                red_start_label_x = take_coord[0] + 10
                red_start_label_y = take_coord[1] + 5
                self.red_number_label[int(coin_number) - 1].place(x=red_start_label_x, y=red_start_label_y)

                if self.red_coin_position[int(coin_number)-1]+self.move_red_counter<=106:
                    self.red_coin_position[int(coin_number)-1] = self.motion_of_coin(self.red_coin_position[int(coin_number) - 1],self.made_red_coin[int(coin_number)-1],self.red_number_label[int(coin_number)-1],red_start_label_x,red_start_label_y,"red",self.move_red_counter, coin_number) 
                        
                    if self.robo_prem and self.red_coin_position[int(coin_number)-1] == 106 and color_coin == "red":
                        self.robo_store.remove(int(coin_number))
                        print("After removing: ", self.robo_store)

                else:
                    if not self.robo_prem: 
                            messagebox.showerror("Not possible","Sorry, not permitted")
                    self.num_btns_state_controller(self.block_value_predict[0][2])

                    if self.robo_prem:
                        robo_operator = "give"
                        self.robo_judge(robo_operator)
                    return

                if  self.red_coin_position[int(coin_number)-1]==22 or self.red_coin_position[int(coin_number)-1]==9 or self.red_coin_position[int(coin_number)-1]==48 or self.red_coin_position[int(coin_number)-1]==35 or self.red_coin_position[int(coin_number)-1]==14 or self.red_coin_position[int(coin_number)-1]==27 or self.red_coin_position[int(coin_number)-1]==40 or self.red_coin_position[int(coin_number)-1]==1:
                    pass
                else:
                    if self.red_coin_position[int(coin_number) - 1] < 100:
                        self.coord_overlap(self.red_coin_position[int(coin_number)-1],color_coin, self.move_red_counter)

                self.red_coord_store[int(coin_number)-1] = self.red_coin_position[int(coin_number)-1]

            else:
                messagebox.showerror("Wrong choice","Sorry, Your coin in not permitted to travel")
                self.num_btns_state_controller(self.block_value_predict[0][2])

                if self.robo_prem == 1:
                    robo_operator = "give"
                    self.robo_judge(robo_operator)
                return
            
            self.current_coin_moved = int(coin_number)

            self.block_value_predict[0][1]['state'] = NORMAL
            
        elif color_coin == "sky_blue":
            self.num_btns_state_controller(self.block_value_predict[1][2], 0)   
            #blue-blue home- red computer
            self.output_li = [self.move_sky_blue_counter, self.sky_blue_coord_store[0], self.sky_blue_coord_store[1], self.sky_blue_coord_store[2], self.sky_blue_coord_store[3],  self.red_coord_store[0], self.red_coord_store[1], self.red_coord_store[2], self.red_coord_store[3] ]    
            
            for i in range(4):
                print("ok")
                #home position check for sky blue coin
                if self.sky_blue_coin_position[i] == -1:
                    self.output_li.append(1)
                else:
                    self.output_li.append(0)


                #home position check for red coin
                if self.red_coin_position[i] == -1:
                    self.output_li.append(1)
                else:
                    self.output_li.append(0)
                
                
                #star position check for sky blue coin
                star_position = [1,9,14,22,27,35,40,48]
                if self.sky_blue_coord_store[i] in star_position:
                    self.output_li.append(1)  
                else:
                    self.output_li.append(0)


                #star position check for red coin
                if self.red_coord_store[i] in star_position:
                    self.output_li.append(1)
                else:
                    self.output_li.append(0)


                #can capture check for red coin
                if self.sky_blue_coord_store[i]+self.move_sky_blue_counter in self.red_coord_store:
                    self.output_li.append(1)
                else:
                    self.output_li.append(0)


                #paka guti for sky blue coin
                if i==0 or i==1:
                    if self.sky_blue_coin_position[i] >= 14 and self.sky_blue_coin_position[i] <= 38:
                        self.output_li.append(1)      
                    else:
                        self.output_li.append(0)
                elif i==2 or i==3:
                    if (self.sky_blue_coin_position[i] >= 40 and self.sky_blue_coin_position[i] <=52) or (self.sky_blue_coin_position[i] >= 1 and self.sky_blue_coin_position[i] <=12):
                        self.output_li.append(1)      
                    else:
                        self.output_li.append(0)


                #paka guti for red coin
                if i==0 or i==1:
                    if self.red_coin_position[i] >= 27 and self.red_coin_position[i] <= 51:
                        self.output_li.append(1)      
                    else:
                        self.output_li.append(0)
                elif i==2 or i==3:
                    if (self.red_coin_position[i] >= 1 and self.red_coin_position[i] <=25):
                        self.output_li.append(1)      
                    else:
                        self.output_li.append(0)


                capture_dis_difference = 100000000000
                for j in range(4):
                    if self.sky_blue_coin_position[i] == -1 or self.sky_blue_coin_position[i] >= 100:
                        capture_dis_difference = -100
                        break
                    if self.red_coin_position[j] == -1 or self.red_coin_position[j] >= 100:
                        capture_dis_difference = 100
                        continue
                    if self.sky_blue_coin_position[i] > self.red_coin_position[j]:
                        capture_dis_difference = min(capture_dis_difference,self.sky_blue_coin_position[i] - self.red_coin_position[j])
                    else:
                        capture_dis_difference = min(capture_dis_difference, 52 - self.red_coin_position[j] + self.sky_blue_coin_position[i])
                self.output_li.append(capture_dis_difference)

                #danger backward blue coin forward red
                dis_difference = 100000000000
                for j in range(4):
                    if self.sky_blue_coin_position[i] == -1 or self.sky_blue_coin_position[i] >= 100:
                        dis_difference = -100
                        break
                    if self.red_coin_position[j] == -1 or self.red_coin_position[j] >= 100:
                        dis_difference = 100
                        continue
                    if self.sky_blue_coin_position[i] < self.red_coin_position[j]:
                        dis_difference = min(dis_difference,self.red_coin_position[j] - self.sky_blue_coin_position[i])
                    else:
                        dis_difference = min(dis_difference, 52 - self.sky_blue_coin_position[i] + self.red_coin_position[j])
                self.output_li.append(dis_difference)
        
                #+n distance
                if self.sky_blue_coin_position[i] == -1:
                    self.output_li.append(-100)
                elif self.sky_blue_coin_position[i] >= 1 and self.sky_blue_coin_position[i] <= 6:
                    self.output_li.append(6-self.sky_blue_coin_position[i])
                elif self.sky_blue_coin_position[i] > 6 and self.sky_blue_coin_position[i] <= 19:
                    self.output_li.append(19-self.sky_blue_coin_position[i])
                elif self.sky_blue_coin_position[i] > 19 and self.sky_blue_coin_position[i] <= 32:
                    self.output_li.append(32-self.sky_blue_coin_position[i])
                elif self.sky_blue_coin_position[i] > 32 and self.sky_blue_coin_position[i] <= 45:
                    self.output_li.append(45-self.sky_blue_coin_position[i])
                elif self.sky_blue_coin_position[i] > 45 and self.sky_blue_coin_position[i] <= 52:
                    self.output_li.append(52-self.sky_blue_coin_position[i]+6)
                else:
                    self.output_li.append(100)

            self.output_li.append(coin_number)
            print(self.output_li)  
            print(len(self.output_li)) 
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

            # Path to the CSV file
            csv_file = os.path.join(desktop_path, "dataset.csv")

            # Append the new data to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([self.output_li])

            if self.move_red_counter == 106:
                messagebox.showwarning("Destination reached","Reached at the destination")

            elif self.sky_blue_coin_position[int(coin_number) - 1] == -1 and self.move_sky_blue_counter == 6:
                if int(coin_number) == 3 or int(coin_number) == 4:
                    self.sky_blue_circle_start_position(coin_number)
                    self.sky_blue_coord_store[int(coin_number) - 1] = 40     # sky blue 3,4 start position
                else:
                    self.green_circle_start_position(coin_number)
                    self.sky_blue_coord_store[int(coin_number) - 1] = 14     # sky blue 1,2 start position

            elif self.sky_blue_coin_position[int(coin_number) - 1] > -1:
                take_coord = self.make_canvas.coords(self.made_sky_blue_coin[int(coin_number) - 1])
                sky_blue_start_label_x = take_coord[0] + 10
                sky_blue_start_label_y = take_coord[1] + 5
                self.sky_blue_number_label[int(coin_number) - 1].place(x=sky_blue_start_label_x, y=sky_blue_start_label_y)

                if  self.sky_blue_coin_position[int(coin_number) - 1] + self.move_sky_blue_counter <= 106:
                    self.sky_blue_coin_position[int(coin_number) - 1] = self.motion_of_coin(self.sky_blue_coin_position[int(coin_number) - 1], self.made_sky_blue_coin[int(coin_number) - 1], self.sky_blue_number_label[int(coin_number) - 1], sky_blue_start_label_x, sky_blue_start_label_y, "sky_blue", self.move_sky_blue_counter, coin_number)
                else:
                   messagebox.showerror("Not possible","No path available")                   
                   self.num_btns_state_controller(self.block_value_predict[1][2])
                   return

                if  self.sky_blue_coin_position[int(coin_number)-1]==22 or self.sky_blue_coin_position[int(coin_number)-1]==9 or self.sky_blue_coin_position[int(coin_number)-1]==48 or self.sky_blue_coin_position[int(coin_number)-1]==35 or self.sky_blue_coin_position[int(coin_number)-1]==1 or self.sky_blue_coin_position[int(coin_number)-1]==14 or self.sky_blue_coin_position[int(coin_number)-1]==27 or self.sky_blue_coin_position[int(coin_number)-1]==40:
                    pass
                else:
                    if self.sky_blue_coin_position[int(coin_number) - 1] < 100:
                        self.coord_overlap(self.sky_blue_coin_position[int(coin_number) - 1],color_coin, self.move_sky_blue_counter)

                self.sky_blue_coord_store[int(coin_number) - 1] = self.sky_blue_coin_position[int(coin_number) - 1]

            else:
                messagebox.showerror("Wrong choice", "Sorry, Your coin in not permitted to travel")
                self.num_btns_state_controller(self.block_value_predict[1][2])
                return
            
            self.current_coin_moved = int(coin_number)

            self.block_value_predict[1][1]['state'] = NORMAL

        print(self.red_coord_store)
        print(self.sky_blue_coord_store)
        if self.robo_prem == 1:
            print("Robo Store is: ", self.robo_store)
        
        permission_granted_to_proceed = True

        if  color_coin == "red" and self.red_coin_position[int(coin_number)-1] == 106:
            permission_granted_to_proceed = self.check_winner_and_runner(color_coin)
        elif  color_coin == "sky_blue" and self.sky_blue_coin_position[int(coin_number)-1] == 106:
            permission_granted_to_proceed = self.check_winner_and_runner(color_coin)

        if permission_granted_to_proceed:       # if that is False, Game is over and not proceed more
            self.make_command(robo_operator)

    def motion_of_coin(self,counter_coin,specific_coin,number_label,number_label_x ,number_label_y,color_coin,path_counter, coin_number):
        try:
            number_label.place(x=number_label_x,y=number_label_y)
            while True:
                if path_counter == 0:
                    break
                elif (counter_coin == 51 and color_coin == "red" and (int(coin_number) == 1 or int(coin_number) == 2)) or (counter_coin==12 and color_coin == "sky_blue" and (int(coin_number) == 1 or int(coin_number) == 2)) or (counter_coin == 25 and color_coin == "red" and (int(coin_number) == 3 or int(coin_number) == 4)) or (counter_coin == 38 and color_coin == "sky_blue" and (int(coin_number) == 3 or int(coin_number) == 4)) or counter_coin>=100:
                    if counter_coin<100:
                        counter_coin=100

                    counter_coin = self.under_room_traversal_control(specific_coin, number_label, number_label_x, number_label_y, path_counter, counter_coin, color_coin, coin_number)

                    if  counter_coin == 106:
                        
                        if self.robo_prem == 1 and color_coin == "red":
                            messagebox.showinfo("Destination reached","Hey! I am at the destination")
                        else:
                            messagebox.showinfo("Destination reached","Congrats! You now at the destination")
                        if path_counter == 6:
                            self.six_with_overlap = 1
                        else:
                            self.time_for -= 1
                    break

                counter_coin += 1
                path_counter -=1
                number_label.place_forget()

                print(counter_coin)

                if counter_coin<=5:
                    self.make_canvas.move(specific_coin, 40, 0)
                    number_label_x+=40
                elif counter_coin == 6:
                    self.make_canvas.move(specific_coin, 40, -40)
                    number_label_x += 40
                    number_label_y-=40
                elif 6< counter_coin <=11:
                    self.make_canvas.move(specific_coin, 0, -40)
                    number_label_y -= 40
                elif counter_coin <=13:
                    self.make_canvas.move(specific_coin, 40, 0)
                    number_label_x += 40
                elif counter_coin <=18:
                    self.make_canvas.move(specific_coin, 0, 40)
                    number_label_y += 40
                elif counter_coin == 19:
                    self.make_canvas.move(specific_coin, 40, 40)
                    number_label_x += 40
                    number_label_y += 40
                elif counter_coin <=24:
                    self.make_canvas.move(specific_coin, 40, 0)
                    number_label_x += 40
                elif counter_coin <=26:
                    self.make_canvas.move(specific_coin, 0, 40)
                    number_label_y += 40
                elif counter_coin <=31:
                    self.make_canvas.move(specific_coin, -40, 0)
                    number_label_x -= 40
                elif counter_coin == 32:
                    self.make_canvas.move(specific_coin, -40, 40)
                    number_label_x -= 40
                    number_label_y += 40
                elif counter_coin <= 37:
                    self.make_canvas.move(specific_coin, 0, 40)
                    number_label_y += 40
                elif counter_coin <= 39:
                    self.make_canvas.move(specific_coin, -40, 0)
                    number_label_x -= 40
                elif counter_coin <= 44:
                    self.make_canvas.move(specific_coin, 0, -40)
                    number_label_y -= 40
                elif counter_coin == 45:
                    self.make_canvas.move(specific_coin, -40, -40)
                    number_label_x -= 40
                    number_label_y -= 40
                elif counter_coin <= 50:
                    self.make_canvas.move(specific_coin, -40, 0)
                    number_label_x -= 40
                elif 50< counter_coin <=52:
                    self.make_canvas.move(specific_coin, 0, -40)
                    number_label_y -= 40
                elif counter_coin == 53:
                    self.make_canvas.move(specific_coin, 40, 0)
                    number_label_x += 40
                    counter_coin = 1

                number_label.place_forget()
                number_label.place(x=number_label_x, y=number_label_y)

                self.window.update()
                time.sleep(0.2)

            return counter_coin
        except:
            print("Force Stop Error Came in motion of coin")

    # For same position, previous coin deleted and set to the room
    def coord_overlap(self, counter_coin, color_coin, path_to_traverse_before_overlap):
        if  color_coin!="red":
            for take_coin_number in range(len(self.red_coord_store)):
                if  self.red_coord_store[take_coin_number] == counter_coin:
                    if path_to_traverse_before_overlap == 6:
                        self.six_with_overlap=1
                    else:
                        self.time_for-=1

                    self.make_canvas.delete(self.made_red_coin[take_coin_number])
                    self.red_number_label[take_coin_number].place_forget()
                    self.red_coin_position[take_coin_number] = -1
                    self.red_coord_store[take_coin_number] = -1
                    if self.robo_prem == 1:
                        self.robo_store.remove(take_coin_number+1)
                        if self.red_coin_position.count(-1)>=1:
                            self.count_robo_stage_from_start = 2

                    if take_coin_number == 0:
                       remade_coin = self.make_canvas.create_oval(100+40, 15+40, 100+40+40, 15+40+40, width=3, fill="red", outline="black")
                       self.red_number_label[take_coin_number].place(x=100 + 40 + 10, y=15 + 40 + 5)
                    elif take_coin_number == 1:
                        remade_coin = self.make_canvas.create_oval(100+40+60+60, 15 + 40, 100+40+60+60+40, 15 + 40 + 40, width=3, fill="red", outline="black")
                        self.red_number_label[take_coin_number].place(x=100 + 40 + 60 +60 + 10, y=15 + 40 + 5)
                    elif take_coin_number == 2:
                        remade_coin = self.make_canvas.create_oval(340 + (40 * 3) + 40 + 60 + 40 + 20, 340 + 80 + 60 + 40 + 15, 340 + (40 * 3) + 40 + 60 + 40 + 40 + 20, 340 + 80 + 60 + 40 + 40 + 15, width=3, fill="red", outline="black")
                        self.red_number_label[take_coin_number].place(x=340+(40*3)+ 40 + 40+ 60 + 30, y=30 + (40*6)+(40*3)+40+100+10)
                    else:
                        remade_coin = self.make_canvas.create_oval(340 + (40 * 3) + 40, 340+80+60+40+15, 340 + (40 * 3) + 40 + 40,340+80+60+40+40+15, width=3,fill="red", outline="black")
                        self.red_number_label[take_coin_number].place(x=340 + (40 * 3) + 40 + 10, y=30 + (40 * 6) + (40 * 3) + 40 + 100 + 10)

                    self.made_red_coin[take_coin_number]=remade_coin

        if  color_coin != "sky_blue":
            for take_coin_number in range(len(self.sky_blue_coord_store)):
                if  self.sky_blue_coord_store[take_coin_number] == counter_coin:
                    if path_to_traverse_before_overlap == 6:
                        self.six_with_overlap = 1
                    else:
                        self.time_for -= 1

                    self.make_canvas.delete(self.made_sky_blue_coin[take_coin_number])
                    self.sky_blue_number_label[take_coin_number].place_forget()
                    self.sky_blue_coin_position[take_coin_number] = -1
                    self.sky_blue_coord_store[take_coin_number]=-1

                    if take_coin_number == 0:
                        remade_coin = self.make_canvas.create_oval(340+(40*3)+40, 15 + 40, 340+(40*3)+40 + 40, 15 + 40 + 40, width=3, fill="#04d9ff", outline="black")
                        self.sky_blue_number_label[take_coin_number].place(x=340 + (40 * 3) + 40 + 10, y=15 + 40 + 5)
                    elif take_coin_number == 1:
                        remade_coin = self.make_canvas.create_oval(340+(40*3)+40+ 60 + 40+20, 15 + 40, 340+(40*3)+40 + 60 + 40 + 40+20, 15 + 40 + 40, width=3, fill="#04d9ff", outline="black")
                        self.sky_blue_number_label[take_coin_number].place(x=340 + (40 * 3) + 40 + 40 + 60 + 30, y=15 + 40 + 5)
                    elif take_coin_number == 2:
                        remade_coin = self.make_canvas.create_oval(100 + 40 + 60 + 40 + 20, 340 + 80 + 60 + 40 + 15, 100 + 40 + 60 + 40 + 40 + 20, 340 + 80 + 60 + 40 + 40 + 15, width=3, fill="#04d9ff", outline="black")
                        self.sky_blue_number_label[take_coin_number].place(x=100 + 40 + 60 + 60 + 10, y=30 + (40 * 6) + (40 * 3) + 40 + 60 + 40 + 10)
                    else:
                        remade_coin = self.make_canvas.create_oval( 100 + 40, 340+80+60+40+15, 100 + 40 + 40, 340+80+60+40+40+15, width=3, fill="#04d9ff", outline="black")
                        self.sky_blue_number_label[take_coin_number].place(x=100+40+10, y=30 + (40*6)+(40*3)+40+60+40+10)

                    self.made_sky_blue_coin[take_coin_number] = remade_coin


    def under_room_traversal_control(self,specific_coin,number_label,number_label_x,number_label_y,path_counter,counter_coin,color_coin, coin_number):
        if color_coin == "red" and counter_coin >= 100 and (int(coin_number) == 1 or int(coin_number) == 2):
            if int(counter_coin)+int(path_counter)<=106:
               counter_coin = self.room_red_traversal(specific_coin, number_label, number_label_x, number_label_y, path_counter, counter_coin)

        elif color_coin == "sky_blue" and counter_coin >= 100 and (int(coin_number) == 1 or int(coin_number) == 2):
            if  int(counter_coin) + int(path_counter) <= 106:
                counter_coin = self.room_green_traversal(specific_coin, number_label, number_label_x, number_label_y,path_counter,counter_coin)

        elif color_coin == "red" and counter_coin >= 100 and (int(coin_number) == 3 or int(coin_number) == 4):
            if  int(counter_coin) + int(path_counter) <= 106:
                counter_coin = self.room_yellow_traversal(specific_coin, number_label, number_label_x, number_label_y,path_counter,counter_coin)

        elif color_coin == "sky_blue" and counter_coin >= 100 and (int(coin_number) == 3 or int(coin_number) == 4):
            if  int(counter_coin) + int(path_counter) <= 106:
                counter_coin = self.room_sky_blue_traversal(specific_coin, number_label, number_label_x, number_label_y,path_counter,counter_coin)

        return counter_coin


    def room_red_traversal(self, specific_coin, number_label, number_label_x, number_label_y, path_counter, counter_coin):
        while path_counter>0:
            counter_coin += 1
            path_counter -= 1
            self.make_canvas.move(specific_coin, 40, 0)
            number_label_x+=40
            number_label.place(x=number_label_x,y=number_label_y)
            self.window.update()
            time.sleep(0.2)
        return counter_coin

    def room_green_traversal(self, specific_coin, number_label, number_label_x, number_label_y, path_counter, counter_coin):
        while path_counter > 0:
            counter_coin += 1
            path_counter -= 1
            self.make_canvas.move(specific_coin, 0, 40)
            number_label_y += 40
            number_label.place(x=number_label_x, y=number_label_y)
            self.window.update()
            time.sleep(0.2)
        return counter_coin

    def room_yellow_traversal(self, specific_coin, number_label, number_label_x, number_label_y,path_counter,counter_coin):
        while path_counter > 0:
            counter_coin += 1
            path_counter -= 1
            self.make_canvas.move(specific_coin, -40, 0)
            number_label_x -= 40
            number_label.place(x=number_label_x, y=number_label_y)
            self.window.update()
            time.sleep(0.2)
        return counter_coin

    def room_sky_blue_traversal(self, specific_coin, number_label, number_label_x, number_label_y,path_counter,counter_coin):
        while path_counter > 0:
            counter_coin += 1
            path_counter -= 1
            self.make_canvas.move(specific_coin, 0, -40)
            number_label_y -= 40
            number_label.place(x=number_label_x, y=number_label_y)
            self.window.update()
            time.sleep(0.2)
        return counter_coin

    def check_winner_and_runner(self,color_coin):
        destination_reached = 0 # Check for all specific color coins
        if color_coin == "red":
            temp_store = self.red_coord_store
        else:
            temp_store = self.sky_blue_coord_store
        
        for take in temp_store:
            if take == 106:
                destination_reached = 1
            else:
                destination_reached = 0
                break

        if  destination_reached == 1:# If all coins in block reach to the destination, winner and runner check
            
            if color_coin == 'red':
                messagebox.showinfo("Computer wins")
            else:
                messagebox.showinfo('Human wins')

            return False
            
        return True

    def model(self):
        
        dataset = pd.read_csv('C:/Users/HP/Desktop/dataset.csv')

        
        X = dataset.drop('choose_piece', axis=1)
        y = dataset['choose_piece']

        print(X.shape)
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        classifier = DecisionTreeClassifier()

        
        classifier.fit(X_train, y_train)

       
        return classifier


    def predict_move(self, predict_li):
        
        # Make predictions on new data
        
        new_prediction = self.classifier.predict([predict_li])
        print(f"New prediction: {new_prediction}")

        return new_prediction[0]


    def robo_judge(self, ind="give"):   
        if ind == "give":# For give the value robot move calculation
            
            predict_li = [self.move_red_counter, self.red_coord_store[0], self.red_coord_store[1], self.red_coord_store[2], self.red_coord_store[3] , self.sky_blue_coord_store[0], self.sky_blue_coord_store[1], self.sky_blue_coord_store[2], self.sky_blue_coord_store[3]]
            
            for i in range(4):
                print("ok")
                #home position check for red coin
                if self.red_coin_position[i] == -1:
                    predict_li.append(1)
                else:
                    predict_li.append(0)

                #home position check for sky blue coin
                if self.sky_blue_coin_position[i] == -1:
                    predict_li.append(1)
                else:
                    predict_li.append(0)
                
                star_position = [1,9,14,22,27,35,40,48]

                if self.red_coord_store[i] in star_position:
                    predict_li.append(1)
                else:
                    predict_li.append(0)
                

                if self.sky_blue_coord_store[i] in star_position:
                    predict_li.append(1)
                    
                else:
                    predict_li.append(0)
                
                
                #can capture check for blue coin
                if self.red_coord_store[i] + self.move_red_counter in self.sky_blue_coord_store:
                    predict_li.append(1)
                else:
                    predict_li.append(0)

                #paka guti for red coin
                if i==0 or i==1:
                    if self.red_coin_position[i] >= 27 and self.red_coin_position[i] <= 51:
                        predict_li.append(1)      
                    else:
                        predict_li.append(0)
                else:
                    if (self.red_coin_position[i] >= 1 and self.red_coin_position[i] <=25):
                        predict_li.append(1)      
                    else:
                        predict_li.append(0)

                #paka guti for sky blue coin
                if i==0 or i==1:
                    if self.sky_blue_coin_position[i] >= 14 and self.sky_blue_coin_position[i] <= 38:
                        predict_li.append(1)      
                    else:
                        predict_li.append(0)
                else:
                    if (self.sky_blue_coin_position[i] >= 40 and self.sky_blue_coin_position[i] <=52) or (self.sky_blue_coin_position[i] >= 1 and self.sky_blue_coin_position[i] <=12):
                        predict_li.append(1)      
                    else:
                        predict_li.append(0)
                
                #danger backward blue coin forward red
                dis_difference = 100000000000
                for j in range(4):
                    if self.red_coin_position[i] == -1 or self.sky_blue_coin_position[i] >= 100:
                        dis_difference = -100
                        break

                    if self.sky_blue_coin_position[j] == -1 or self.sky_blue_coin_position[j] >= 100:
                        dis_difference = 100
                        continue

                    if self.sky_blue_coin_position[j] < self.red_coin_position[i]:
                        dis_difference = min(dis_difference,self.red_coin_position[i] - self.sky_blue_coin_position[j])
                    else:
                        dis_difference = min(dis_difference, 52 - self.sky_blue_coin_position[j] + self.red_coin_position[i])
                predict_li.append(dis_difference)
                
                capture_dis_difference = 100000000000
                for j in range(4):
                    if self.red_coin_position[i] == -1 or self.red_coin_position[i] >= 100:
                        capture_dis_difference = -100
                        break
                    if self.sky_blue_coin_position[j] == -1 or self.sky_blue_coin_position[j] >= 100:
                        capture_dis_difference = 100
                        continue
                    if self.sky_blue_coin_position[j] > self.red_coin_position[i]:
                        capture_dis_difference = min(capture_dis_difference,self.sky_blue_coin_position[i] - self.red_coin_position[j])
                    else:
                        capture_dis_difference = min(capture_dis_difference, 52 - self.red_coin_position[i] + self.sky_blue_coin_position[j])
                predict_li.append(capture_dis_difference)
                
                #+n distance
                if self.red_coin_position[i] == -1:
                    predict_li.append(-100)
                elif self.red_coin_position[i] >= 1 and self.red_coin_position[i] <= 6:
                    predict_li.append(6-self.red_coin_position[i])
                elif self.red_coin_position[i] > 6 and self.red_coin_position[i] <= 19:
                    predict_li.append(19-self.red_coin_position[i])
                elif self.red_coin_position[i] > 19 and self.red_coin_position[i] <= 32:
                    predict_li.append(32-self.red_coin_position[i])
                elif self.red_coin_position[i] > 32 and self.red_coin_position[i] <= 45:
                    predict_li.append(45-self.red_coin_position[i])
                elif self.red_coin_position[i] > 45 and self.red_coin_position[i] <= 52:
                    predict_li.append(52-self.red_coin_position[i]+6)
                else:
                    predict_li.append(100)

            print(predict_li)
            print("predict_li ",len(predict_li))
            
            prd_value = self.predict_move(predict_li)

            if self.red_coin_position[self.current_coin_moved-1] == 6 or  self.red_coin_position[self.current_coin_moved-1] == 19 or self.red_coin_position[self.current_coin_moved-1] == 32 or self.red_coin_position[self.current_coin_moved-1] == 45: # hello_dice_choice
                prd_value = self.current_coin_moved

            print("prd_model ",prd_value)
            all_in = 1      # Denoting all the coins are present in the room
            for i in range(4):
                if self.red_coin_position[i] == -1:
                    all_in = 1
                else:
                    all_in = 0      # Denoting all the coins not present in the room
                    break

            if all_in == 1:
                if self.move_red_counter == 6:
                    self.robo_store.append(prd_value)
                    self.main_controller("red", prd_value)
                else:
                    pass
            else:
                if prd_value not in self.robo_store:
                    if self.move_red_counter == 6:
                        self.robo_store.append(prd_value)
                        self.main_controller("red", prd_value)
                    else:
                        prd_value = self.robo_store[randint(0, len(self.robo_store)-1)]
                        self.main_controller("red", prd_value)
                else:
                    self.main_controller("red", prd_value)

            print("final_prd_model ",prd_value)
            print(self.red_coin_position[prd_value-1])
            if self.red_coin_position[prd_value-1] == 6 or self.red_coin_position[prd_value-1] == 19 or self.red_coin_position[prd_value-1] == 32 or self.red_coin_position[prd_value-1] == 45: # hello_dice_choice
                self.choice_dice = True
                self.make_prediction("red")
            
            
        else:
            if self.sky_blue_coord_store[self.current_coin_moved-1] == 6 or  self.sky_blue_coord_store[self.current_coin_moved-1] == 19 or self.sky_blue_coord_store[self.current_coin_moved-1] == 32 or self.sky_blue_coord_store[self.current_coin_moved-1] == 45: # hello_dice_choice
                self.choice_dice = True
                self.make_prediction("sky_blue")
            else:
                self.make_prediction("red")     # Prediction Function Call


if __name__ == '__main__':
    window = Tk()
    window.geometry("800x630")
    window.maxsize(800,630)
    window.minsize(800,630)
    window.title("Ludo 2.0")
    block_six_side = ImageTk.PhotoImage(Image.open("Images/6_block.png").resize((33, 33), Image.ANTIALIAS))
    block_five_side = ImageTk.PhotoImage(Image.open("Images/5_block.png").resize((33, 33), Image.ANTIALIAS))
    block_four_side = ImageTk.PhotoImage(Image.open("Images/4_block.png").resize((33, 33), Image.ANTIALIAS))
    block_three_side = ImageTk.PhotoImage(Image.open("Images/3_block.png").resize((33, 33), Image.ANTIALIAS))
    block_two_side = ImageTk.PhotoImage(Image.open("Images/2_block.png").resize((33, 33), Image.ANTIALIAS))
    block_one_side = ImageTk.PhotoImage(Image.open("Images/1_block.png").resize((33, 33), Image.ANTIALIAS))
    Ludo(window,block_six_side,block_five_side,block_four_side,block_three_side,block_two_side,block_one_side)
    window.mainloop()
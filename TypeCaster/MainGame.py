import pygame as pg
import random
import time
import sys
import os
import math
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import glob

pg.init()

# ============================
# ENHANCED WORD SYSTEM WITH MULTIPLE FILES
# ============================

class WordLibrary:
    def __init__(self):
        self.categories = {}
        self.word_difficulty = {}  # word -> difficulty score (1-10)
        self.all_words = []
        self.load_all_word_files()
        self.categorize_words()
        
    def load_all_word_files(self):
        """Load words from multiple themed files"""
        word_file_options = [
            "spells.txt", "words_common.txt", "words_tech.txt", 
            "words_fantasy.txt", "words_difficult.txt", "words_short.txt", "words_long.txt"
        ]
        
        self.all_words = []
        loaded_files = []
        
        for filename in word_file_options:
            try:
                if os.path.exists(filename):
                    with open(filename, "r", encoding='utf-8') as f:
                        words = [line.strip() for line in f if line.strip()]
                        words = [w for w in words if len(w) <= 25]
                        self.all_words.extend(words)
                        loaded_files.append(filename)
                        print(f"Loaded {len(words)} words from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        if not self.all_words:
            print("No word files found. Creating default word list...")
            self.all_words = [
                "hello", "world", "python", "game", "type", "speed", "test",
                "keyboard", "mouse", "screen", "player", "enemy", "attack"
            ]
            loaded_files.append("default_words")
        
        print(f"Total words loaded: {len(self.all_words)} from {len(loaded_files)} files")
        
    def categorize_words(self):
        """Categorize words by various properties"""
        if not self.all_words:
            return
            
        self.categories = {
            'short': [], 'medium': [], 'long': [],
            'left_hand_dominant': [], 'right_hand_dominant': [],
            'home_row': [], 'top_row': [], 'bottom_row': [],
            'repeated_letters': [], 'rare_letters': [],
            'vowel_heavy': [], 'consonant_heavy': [],
            'common': [], 'tech': [], 'fantasy': [], 'difficult': [],
            'alternating_hands': [],
        }
        
        for word in self.all_words:
            word_lower = word.lower()
            length = len(word_lower)
            
            if length <= 4:
                self.categories['short'].append(word)
            elif 5 <= length <= 7:
                self.categories['medium'].append(word)
            elif length <= 15:
                self.categories['long'].append(word)
            
            if self._is_left_hand_dominant(word_lower):
                self.categories['left_hand_dominant'].append(word)
            if self._is_right_hand_dominant(word_lower):
                self.categories['right_hand_dominant'].append(word)
            if self._is_home_row(word_lower):
                self.categories['home_row'].append(word)
            if self._is_top_row(word_lower):
                self.categories['top_row'].append(word)
            if self._is_bottom_row(word_lower):
                self.categories['bottom_row'].append(word)
            
            if self._has_repeated_letters(word_lower):
                self.categories['repeated_letters'].append(word)
            if self._has_rare_letters(word_lower):
                self.categories['rare_letters'].append(word)
            if self._vowel_ratio(word_lower) > 0.6:
                self.categories['vowel_heavy'].append(word)
            if self._vowel_ratio(word_lower) < 0.3:
                self.categories['consonant_heavy'].append(word)
            
            if self._has_alternating_hands(word_lower):
                self.categories['alternating_hands'].append(word)
        
        self._calculate_word_difficulties()
        
        for category in list(self.categories.keys()):
            if not self.categories[category]:
                self.categories[category] = random.sample(self.all_words, min(10, len(self.all_words)))
        
        print(f"Words categorized into {len(self.categories)} categories")
    
    def _is_left_hand_dominant(self, word):
        left_hand = set('qwertasdfgzxcvb')
        left_count = sum(1 for c in word if c in left_hand)
        return left_count / len(word) > 0.7
    
    def _is_right_hand_dominant(self, word):
        right_hand = set('yuiophjklmn')
        right_count = sum(1 for c in word if c in right_hand)
        return right_count / len(word) > 0.7
    
    def _is_home_row(self, word):
        home_row = set('asdfghjkl')
        home_count = sum(1 for c in word if c in home_row)
        return home_count / len(word) > 0.7
    
    def _is_top_row(self, word):
        top_row = set('qwertyuiop')
        top_count = sum(1 for c in word if c in top_row)
        return top_count / len(word) > 0.7
    
    def _is_bottom_row(self, word):
        bottom_row = set('zxcvbnm')
        bottom_count = sum(1 for c in word if c in bottom_row)
        return bottom_count / len(word) > 0.7
    
    def _has_repeated_letters(self, word):
        return any(word[i] == word[i+1] for i in range(len(word)-1))
    
    def _has_rare_letters(self, word):
        rare_letters = set('zqxjkv')
        return any(c in rare_letters for c in word)
    
    def _vowel_ratio(self, word):
        vowels = set('aeiouy')
        vowel_count = sum(1 for c in word if c in vowels)
        return vowel_count / len(word) if word else 0
    
    def _has_alternating_hands(self, word):
        left_hand = set('qwertasdfgzxcvb')
        if len(word) < 2:
            return False
        
        first_in_left = word[0] in left_hand
        for i in range(1, len(word)):
            current_in_left = word[i] in left_hand
            if current_in_left == first_in_left:
                return False
            first_in_left = current_in_left
        return True
    
    def _calculate_word_difficulties(self):
        for word in self.all_words:
            word_lower = word.lower()
            score = 1
            
            length = len(word_lower)
            if length <= 4:
                score += 1
            elif length <= 6:
                score += 2
            elif length <= 8:
                score += 3
            elif length <= 12:
                score += 4
            else:
                score += 5
            
            rare_letters = set('zqxjkv')
            score += sum(1 for c in word_lower if c in rare_letters)
            
            if self._has_repeated_letters(word_lower):
                score += 1
            
            if self._has_alternating_hands(word_lower):
                score = max(1, score - 1)
            
            if any(not c.isalpha() for c in word):
                score += 2
            
            self.word_difficulty[word] = min(10, max(1, score))
    
    def get_words_by_difficulty(self, min_difficulty=1, max_difficulty=10):
        return [w for w, d in self.word_difficulty.items() 
                if min_difficulty <= d <= max_difficulty]
    
    def get_words_by_category(self, category):
        return self.categories.get(category, self.all_words)
    
    def get_random_word(self, category=None, difficulty_range=None):
        word_pool = self.all_words
        
        if category and category in self.categories:
            word_pool = self.categories[category]
        
        if difficulty_range:
            filtered = [w for w in word_pool 
                       if difficulty_range[0] <= self.word_difficulty.get(w, 5) <= difficulty_range[1]]
            if len(filtered) < 5:
                expanded_min = max(1, difficulty_range[0] - 2)
                expanded_max = min(10, difficulty_range[1] + 1)
                filtered = [w for w in word_pool 
                           if expanded_min <= self.word_difficulty.get(w, 5) <= expanded_max]
            
            if filtered:
                word_pool = filtered
        
        if not word_pool:
            word_pool = self.all_words
        
        displayable_words = [w for w in word_pool if len(w) <= 20]
        if displayable_words:
            return random.choice(displayable_words)
        
        short_words = [w for w in self.all_words if len(w) <= 15]
        if short_words:
            return random.choice(short_words)
        
        return random.choice(self.all_words)


# ============================
# MACHINE LEARNING MODEL
# ============================

class TypingMLPredictor:
    """Machine Learning model to predict typing success"""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.trained = False
        self.metrics = {}
        self.feature_names = [
            'prev_correct', 'prev_time', 'prev_difficulty',
            'current_length', 'current_difficulty',
            'rolling_accuracy', 'avg_wpm'
        ]
    
    def load_all_json_data(self, folder_path="."):
        """Load all JSON files from the current directory"""
        all_data = []
        json_pattern = os.path.join(folder_path, "*.json")
        json_files = glob.glob(json_pattern)
        
        print(f"Found {len(json_files)} JSON files for ML training")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return all_data
    
    def prepare_training_data(self, game_data):
        """Convert game logs into ML features and labels"""
        X_features = []
        y_labels = []
        
        attempts = [d for d in game_data if d.get('type') == 'word_attempt']
        
        for i in range(3, len(attempts)):
            prev_three = attempts[i-3:i]
            
            prev_correct = 1 if prev_three[-1]['data']['correct'] else 0
            prev_time = prev_three[-1]['data']['time_taken']
            prev_difficulty = prev_three[-1]['data']['word_difficulty']
            
            current_word = attempts[i]['data']['word']
            current_length = len(current_word)
            current_difficulty = attempts[i]['data']['word_difficulty']
            
            correct_in_last_three = sum(1 for a in prev_three if a['data']['correct'])
            rolling_accuracy = correct_in_last_three / 3.0
            
            total_time = sum(a['data']['time_taken'] for a in prev_three)
            if total_time > 0:
                total_chars = sum(len(a['data']['word']) for a in prev_three)
                avg_wpm = (total_chars / 5) / (total_time / 60)
            else:
                avg_wpm = 0
            
            features = [
                prev_correct, prev_time, prev_difficulty,
                current_length, current_difficulty,
                rolling_accuracy, avg_wpm
            ]
            
            label = 1 if attempts[i]['data']['correct'] else 0
            
            X_features.append(features)
            y_labels.append(label)
        
        return np.array(X_features), np.array(y_labels)
    
    def train_model(self, json_folder="."):
        """Train the ML model and evaluate performance"""
        all_data = self.load_all_json_data(json_folder)
        
        if len(all_data) < 20:
            print("Not enough data to train ML model. Need more game sessions.")
            return False
        
        X, y = self.prepare_training_data(all_data)
        
        if len(X) < 10:
            print(f"Only {len(X)} training samples. Need at least 10.")
            return False
        
        print(f"Training ML model with {len(X)} samples...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'baseline_accuracy': max(np.mean(y_test), 1 - np.mean(y_test)),
            'feature_importance': dict(zip(self.feature_names, self.model.coef_[0]))
        }
        
        self.trained = True
        self.print_model_performance()
        self.save_model_info()
        
        return True
    
    def print_model_performance(self):
        """Display model performance metrics"""
        print("\n" + "="*60)
        print("MACHINE LEARNING MODEL PERFORMANCE")
        print("="*60)
        print(f"Model: Logistic Regression")
        print(f"Training samples: {self.metrics['training_samples']}")
        print(f"Test samples: {self.metrics['test_samples']}")
        print(f"Accuracy: {self.metrics['accuracy']:.1%}")
        print(f"Precision: {self.metrics['precision']:.1%}")
        print(f"Recall: {self.metrics['recall']:.1%}")
        print(f"F1-Score: {self.metrics['f1_score']:.1%}")
        print(f"Baseline Accuracy: {self.metrics['baseline_accuracy']:.1%}")
        print(f"Improvement over baseline: {self.metrics['accuracy'] - self.metrics['baseline_accuracy']:.1%}")
        
        print("\nConfusion Matrix:")
        cm = self.metrics['confusion_matrix']
        print(f"          Predicted")
        print(f"          Correct  Wrong")
        print(f"Actual Correct  {cm[1][1]:3d}     {cm[1][0]:3d}")
        print(f"       Wrong     {cm[0][1]:3d}     {cm[0][0]:3d}")
        
        print("\nFeature Importance (higher = more predictive):")
        for feature, importance in sorted(self.metrics['feature_importance'].items(), 
                                         key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:20s}: {importance:+.3f}")
        print("="*60 + "\n")
    
    def save_model_info(self, filename="ml_model_report.txt"):
        """Save model information to a file for presentation"""
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MACHINE LEARNING MODEL REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Algorithm: Logistic Regression\n")
            f.write(f"Training Samples: {self.metrics['training_samples']}\n")
            f.write(f"Test Samples: {self.metrics['test_samples']}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {self.metrics['accuracy']:.1%}\n")
            f.write(f"Precision: {self.metrics['precision']:.1%}\n")
            f.write(f"Recall: {self.metrics['recall']:.1%}\n")
            f.write(f"F1-Score: {self.metrics['f1_score']:.1%}\n")
            f.write(f"Baseline Accuracy: {self.metrics['baseline_accuracy']:.1%}\n")
            f.write(f"Improvement: {self.metrics['accuracy'] - self.metrics['baseline_accuracy']:.1%}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 40 + "\n")
            cm = self.metrics['confusion_matrix']
            f.write(f"           Predicted\n")
            f.write(f"           Correct  Wrong\n")
            f.write(f"Actual Correct  {cm[1][1]:3d}     {cm[1][0]:3d}\n")
            f.write(f"       Wrong     {cm[0][1]:3d}     {cm[0][0]:3d}\n\n")
            
            f.write("FEATURE IMPORTANCE:\n")
            f.write("-" * 40 + "\n")
            for feature, importance in sorted(self.metrics['feature_importance'].items(), 
                                            key=lambda x: abs(x[1]), reverse=True):
                f.write(f"{feature:20s}: {importance:+.3f}\n")
        
        print(f"ML report saved to {filename}")
    
    def predict_success_probability(self, features):
        """Predict probability of typing success (0-1)"""
        if not self.trained:
            return 0.5
        
        features_array = np.array(features).reshape(1, -1)
        prob = self.model.predict_proba(features_array)[0][1]
        return prob
    
    def get_optimal_difficulty(self, player_stats):
        """Use ML to suggest optimal next word difficulty"""
        if not self.trained:
            return 5
        
        success_prob = self.predict_success_probability(player_stats)
        optimal_difficulty = max(1, min(10, int(1 + (success_prob * 9))))
        return optimal_difficulty


# ============================
# AI SYSTEM WITH PROGRESSIVE DIFFICULTY
# ============================

class PatternExploiterAI:
    def __init__(self, word_library):
        self.library = word_library
        self.player_profile = {
            'mistakes_by_letter': {},
            'mistakes_by_category': {},
            'average_times': {},
            'difficulty_success_rate': {},
            'recent_words': [],
            'caps_lock_errors': 0,
        }
        self.personality = random.choice(['aggressive', 'defensive', 'tricky', 'balanced'])
        print(f"AI Personality: {self.personality}")
        
        self.game_difficulty = 1
        self.consecutive_correct = 0
        self.consecutive_failures = 0
        self.average_wpm_history = []
        self.accuracy_history = []
        
    def record_typing_event(self, word, typed_word, correct, time_taken):
        word_lower = word.lower()
        typed_lower = typed_word.lower()
        
        self.player_profile['recent_words'].append(word)
        if len(self.player_profile['recent_words']) > 10:
            self.player_profile['recent_words'].pop(0)
        
        if word != typed_word and word_lower == typed_lower:
            self.player_profile['caps_lock_errors'] += 1
        
        wrong_letters = []
        min_len = min(len(typed_lower), len(word_lower))
        for i in range(min_len):
            if typed_lower[i] != word_lower[i]:
                wrong_letters.append(word_lower[i])
        
        for letter in wrong_letters:
            self.player_profile['mistakes_by_letter'][letter] = \
                self.player_profile['mistakes_by_letter'].get(letter, 0) + 1
        
        categories = self._get_word_categories(word_lower)
        difficulty = self.library.word_difficulty.get(word, 5)
        
        for category in categories:
            if category not in self.player_profile['mistakes_by_category']:
                self.player_profile['mistakes_by_category'][category] = 0
                self.player_profile['average_times'][category] = []
            
            if not correct:
                self.player_profile['mistakes_by_category'][category] += 1
            
            self.player_profile['average_times'][category].append(time_taken)
            if len(self.player_profile['average_times'][category]) > 5:
                self.player_profile['average_times'][category].pop(0)
        
        if difficulty not in self.player_profile['difficulty_success_rate']:
            self.player_profile['difficulty_success_rate'][difficulty] = {'success': 0, 'total': 0}
        
        self.player_profile['difficulty_success_rate'][difficulty]['total'] += 1
        if correct:
            self.player_profile['difficulty_success_rate'][difficulty]['success'] += 1
        
        if correct:
            self.consecutive_correct += 1
            self.consecutive_failures = max(0, self.consecutive_failures - 1)
        else:
            self.consecutive_failures += 1
            self.consecutive_correct = max(0, self.consecutive_correct - 2)
        
        if time_taken > 0:
            wpm_for_word = len(word) / 5 / (time_taken / 60)
            self.average_wpm_history.append(wpm_for_word)
            if len(self.average_wpm_history) > 10:
                self.average_wpm_history.pop(0)
    
    def _get_word_categories(self, word):
        word_lower = word.lower()
        categories = []
        for category, words in self.library.categories.items():
            if any(w.lower() == word_lower for w in words):
                categories.append(category)
        return categories
    
    def _get_player_weakness(self):
        if self.player_profile['caps_lock_errors'] > 3:
            return 'case_sensitive'
        
        weakness_scores = {}
        for category, mistake_count in self.player_profile['mistakes_by_category'].items():
            total_attempts = len(self.player_profile['average_times'].get(category, []))
            if total_attempts >= 3:
                weakness_scores[category] = mistake_count / total_attempts
        
        if weakness_scores:
            return max(weakness_scores.items(), key=lambda x: x[1])[0]
        
        if self.player_profile['mistakes_by_letter']:
            worst_letter = max(self.player_profile['mistakes_by_letter'].items(), 
                             key=lambda x: x[1])[0]
            if worst_letter in 'qwertasdfgzxcvb':
                return 'left_hand_dominant'
            else:
                return 'right_hand_dominant'
        
        return None
    
    def _get_optimal_difficulty(self):
        if not self.player_profile['difficulty_success_rate']:
            return (max(1, self.game_difficulty - 2), min(10, self.game_difficulty + 2))
        
        base_min = max(1, self.game_difficulty - 1)
        base_max = min(10, self.game_difficulty + 2)
        
        if self.consecutive_correct >= 5:
            adjustment = min(3, self.consecutive_correct // 3)
            base_min = min(10, base_min + adjustment)
            base_max = min(10, base_max + adjustment)
        elif self.consecutive_failures >= 3:
            adjustment = min(2, self.consecutive_failures // 2)
            base_min = max(1, base_min - adjustment)
            base_max = max(3, base_max - adjustment)
        
        return (base_min, base_max)
    
    def update_game_difficulty(self, wpm, accuracy, time_since_start):
        wpm_score = min(100, wpm * 1.5)
        accuracy_score = accuracy
        
        performance_score = (wpm_score * 0.4) + (accuracy_score * 0.6)
        
        time_factor = min(5, time_since_start / 60)
        
        new_difficulty = 1 + (performance_score / 15) + time_factor
        
        self.game_difficulty = min(10, max(1, new_difficulty))
        
        if self.game_difficulty > 7:
            self.personality = 'aggressive'
        elif self.game_difficulty > 5:
            self.personality = 'tricky'
        
        return self.game_difficulty
    
    def choose_challenging_word(self):
        weakness = self._get_player_weakness()
        difficulty_range = self._get_optimal_difficulty()
        
        if self.game_difficulty >= 9:
            difficulty_range = (max(7, difficulty_range[0]), 10)
        
        if self.personality == 'aggressive':
            difficulty_range = (min(10, difficulty_range[0] + 1), 
                              min(10, difficulty_range[1] + 1))
            if weakness == 'case_sensitive':
                word = self.library.get_random_word(difficulty_range=difficulty_range)
                return self._randomize_case(word, self.game_difficulty)
            elif weakness:
                return self.library.get_random_word(weakness, difficulty_range)
            else:
                if self.game_difficulty > 6:
                    if random.random() > 0.6:
                        return self.library.get_random_word('long', difficulty_range)
                    else:
                        return self.library.get_random_word(difficulty_range=difficulty_range)
                else:
                    return self.library.get_random_word(difficulty_range=difficulty_range)
        
        elif self.personality == 'defensive':
            difficulty_range = (max(1, difficulty_range[0] - 1), 
                              max(3, difficulty_range[1] - 1))
            return self.library.get_random_word('medium', difficulty_range)
        
        elif self.personality == 'tricky':
            categories = ['repeated_letters', 'rare_letters', 'consonant_heavy']
            if self.game_difficulty > 6:
                categories.extend(['top_row', 'bottom_row', 'alternating_hands'])
            
            category = random.choice(categories)
            word = self.library.get_random_word(category, difficulty_range)
            if random.random() > (0.7 - (self.game_difficulty * 0.03)):
                word = self._randomize_case(word, self.game_difficulty)
            return word
        
        else:
            if weakness == 'case_sensitive':
                word = self.library.get_random_word(difficulty_range=difficulty_range)
                return self._randomize_case(word, self.game_difficulty)
            elif weakness and random.random() > 0.3:
                return self.library.get_random_word(weakness, difficulty_range)
            else:
                return self.library.get_random_word(difficulty_range=difficulty_range)
    
    def choose_counter_word(self):
        difficulty_range = self._get_optimal_difficulty()
        
        difficulty_boost = min(3, self.game_difficulty // 2)
        difficulty_range = (min(10, difficulty_range[0] + difficulty_boost), 
                          min(10, difficulty_range[1] + difficulty_boost + 1))
        
        if difficulty_range[0] >= 9 and difficulty_range[1] >= 10:
            difficulty_range = (8, 10)
        
        tricky_categories = ['repeated_letters', 'rare_letters', 'long', 
                           'consonant_heavy', 'top_row', 'bottom_row']
        
        if self.game_difficulty > 7:
            tricky_categories.extend(['alternating_hands', 'vowel_heavy'])
        
        category = random.choice(tricky_categories)
        word = self.library.get_random_word(category, difficulty_range)
        
        if len(word) > 20:
            shorter_words = [w for w in self.library.get_words_by_category(category) 
                           if len(w) <= 15 and difficulty_range[0] <= self.library.word_difficulty.get(w, 5) <= difficulty_range[1]]
            if shorter_words:
                word = random.choice(shorter_words)
        
        case_chance = 0.3 + (self.game_difficulty * 0.05)
        if random.random() < case_chance:
            word = self._randomize_case(word, self.game_difficulty)
        
        return word
    
    def _randomize_case(self, word, difficulty_level):
        if len(word) <= 1:
            return word.upper() if random.random() > 0.5 else word.lower()
        
        if difficulty_level > 8:
            pattern = random.choice(['alternating', 'random', 'camel', 'inverse'])
            
            if pattern == 'alternating':
                result = []
                upper = random.choice([True, False])
                for char in word:
                    if char.isalpha():
                        result.append(char.upper() if upper else char.lower())
                        upper = not upper
                    else:
                        result.append(char)
                return ''.join(result)
                
            elif pattern == 'camel':
                result = []
                next_upper = False
                for char in word:
                    if char.isalpha():
                        result.append(char.upper() if next_upper else char.lower())
                        next_upper = True
                    else:
                        result.append(char)
                        next_upper = False
                return ''.join(result)
                
            elif pattern == 'inverse':
                return word.swapcase()
                
            else:
                result = []
                for char in word:
                    if char.isalpha() and random.random() > 0.5:
                        result.append(char.upper() if char.islower() else char.lower())
                    else:
                        result.append(char)
                return ''.join(result)
        
        elif difficulty_level > 5:
            result = []
            for char in word:
                if char.isalpha() and random.random() > 0.7:
                    result.append(char.upper() if char.islower() else char.lower())
                else:
                    result.append(char)
            return ''.join(result)
        
        else:
            r = random.random()
            if r < 0.4:
                return word.upper()
            elif r < 0.8:
                return word.lower()
            else:
                result = []
                for char in word:
                    if char.isalpha() and random.random() > 0.8:
                        result.append(char.upper() if char.islower() else char.lower())
                    else:
                        result.append(char)
                return ''.join(result)
    
    def get_counter_time_based_on_difficulty(self, word):
        word_difficulty = self.library.word_difficulty.get(word, 5)
        
        base_time = max(1.5, 5.0 - (self.game_difficulty * 0.35))
        
        word_time_reduction = (word_difficulty - 5) * 0.2
        
        counter_time = max(1.0, min(5.0, base_time - word_time_reduction))
        return counter_time
    
    def get_normal_word_time(self):
        base_time = max(5.0, 12.0 - (self.game_difficulty * 0.7))
        return base_time
    
    def should_increase_attack_frequency(self):
        base_chance = 0.3 + (self.game_difficulty * 0.05)
        return random.random() < base_chance


class GameDataCollector:
    def __init__(self):
        self.session_data = []
        self.start_time = time.time()
    
    def record_event(self, event_type, **data):
        event = {
            'type': event_type,
            'timestamp': time.time() - self.start_time,
            'data': data
        }
        self.session_data.append(event)
    
    def save_session(self, filename=None):
        if not filename:
            filename = f"session_{int(time.time())}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            print(f"Session data saved to {filename}")
        except Exception as e:
            print(f"Error saving session data: {e}")
        
        return filename


# ============================
# CREATE SAMPLE WORD FILES IF THEY DON'T EXIST
# ============================

def create_sample_word_files():
    if not os.path.exists("words_common.txt"):
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
        ]
        with open("words_common.txt", "w") as f:
            f.write("\n".join(common_words))
        print("Created words_common.txt")
    
    if not os.path.exists("words_tech.txt"):
        tech_words = [
            "algorithm", "binary", "cache", "database", "encryption",
            "firewall", "gigabyte", "hardware", "interface", "javascript"
        ]
        with open("words_tech.txt", "w") as f:
            f.write("\n".join(tech_words))
        print("Created words_tech.txt")
    
    if not os.path.exists("words_fantasy.txt"):
        fantasy_words = [
            "dragon", "wizard", "sword", "shield", "castle",
            "knight", "magic", "spell", "potion", "dungeon"
        ]
        with open("words_fantasy.txt", "w") as f:
            f.write("\n".join(fantasy_words))
        print("Created words_fantasy.txt")
    
    if not os.path.exists("words_difficult.txt"):
        difficult_words = [
            "antidisestablishmentarianism", "pneumonoultramicroscopicsilicovolcanoconiosis",
            "supercalifragilisticexpialidocious", "floccinaucinihilipilification"
        ]
        difficult_words = [w for w in difficult_words if len(w) <= 30]
        with open("words_difficult.txt", "w") as f:
            f.write("\n".join(difficult_words))
        print("Created words_difficult.txt")
    
    if not os.path.exists("words_short.txt"):
        short_words = [
            "a", "I", "am", "be", "to", "of", "in", "it", "is", "on",
            "he", "as", "at", "by", "we", "or", "an", "my", "up", "if"
        ]
        with open("words_short.txt", "w") as f:
            f.write("\n".join(short_words))
        print("Created words_short.txt")
    
    if not os.path.exists("words_long.txt"):
        long_words = [
            "extraordinary", "unbelievable", "responsibility", "communication",
            "understanding", "international", "organization", "development"
        ]
        with open("words_long.txt", "w") as f:
            f.write("\n".join(long_words))
        print("Created words_long.txt")

create_sample_word_files()

# ============================
# INITIALIZE ENHANCED SYSTEMS
# ============================

word_lib = WordLibrary()
enemy_ai = PatternExploiterAI(word_lib)
data_collector = GameDataCollector()

# Initialize ML Predictor
ml_predictor = TypingMLPredictor()

# Train ML model on existing JSON files
print("\n" + "="*60)
print("TRAINING MACHINE LEARNING MODEL")
print("="*60)
ml_trained = ml_predictor.train_model()
if not ml_trained:
    print("ML model training failed or insufficient data. Using default rules.")

# ============================
# REST OF YOUR SETTINGS
# ============================

WIDTH, HEIGHT = 1050, 540
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("TypeCaster Battle Prototype - ML Enhanced")
clock = pg.time.Clock()
FPS = 60

def safe_load(path, fallback_size=None):
    try:
        return pg.image.load(path).convert_alpha()
    except Exception:
        if fallback_size:
            s = pg.Surface(fallback_size, pg.SRCALPHA)
            s.fill((60, 60, 60, 255))
            return s
        raise

battle_bg = safe_load("battle_bg.jpg", (WIDTH, HEIGHT))
enemy_frame = safe_load("enemy_frame.png", (260, 62))
player_frame = safe_load("player_frame.png", (260, 86))
textbox_bg = safe_load("textbox_bg.png", (WIDTH, 140))

enemy_sprite_img = safe_load("pokemaniac.png", (140, 140))
player_sprite_img = safe_load("pokemonb.png", (140, 140))
enemy_sprite_img = pg.transform.scale(enemy_sprite_img, (200, 200))
player_sprite_img = pg.transform.scale(player_sprite_img, (300, 300))

enemy_sprite_grayscale = enemy_sprite_img.copy()
pixels = pg.surfarray.array3d(enemy_sprite_grayscale)
grayscale = (0.21 * pixels[:,:,0] + 0.72 * pixels[:,:,1] + 0.07 * pixels[:,:,2])
pixels[:,:,0] = grayscale
pixels[:,:,1] = grayscale
pixels[:,:,2] = grayscale
pg.surfarray.blit_array(enemy_sprite_grayscale, pixels)

font_big = pg.font.Font("ByteBounce.TTF", 42) if os.path.exists("ByteBounce.TTF") else pg.font.SysFont(None, 42)
font_med = pg.font.Font("ByteBounce.TTF", 26) if os.path.exists("ByteBounce.TTF") else pg.font.SysFont(None, 26)
font_small = pg.font.Font("ByteBounce.TTF", 20) if os.path.exists("ByteBounce.TTF") else pg.font.SysFont(None, 20)

WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (80,240,80)
YELLOW = (240,200,40)
RED = (240,60,60)
PURPLE = (180, 80, 220)
CYAN = (80, 200, 220)
GRAY = (100, 100, 100)
DARK_GRAY = (60, 60, 60)
ML_COLOR = (255, 165, 0)  # Orange for ML display

# ============================
# GAME STATE
# ============================
NORMAL_PHASE = "normal"
ENEMY_ATTACK_PHASE = "enemy_attack"
game_phase = NORMAL_PHASE

words_typed_this_turn = 0
WORDS_PER_TURN = 3
enemy_attack_word = ""
counter_timer = 0.0
counter_success = False
counter_failed = False

game_over = False
game_over_fade = 0.0
enemy_hp_max = 100
player_hp_max = 100
enemy_hp = enemy_hp_max
player_hp = player_hp_max

enemy_dead = False
enemy_death_timer = 0
ENEMY_RESURRECTION_TIME = 180
enemy_resurrection_progress = 0

disp_enemy_hp = float(enemy_hp)
disp_player_hp = float(player_hp)

current_word = enemy_ai.choose_challenging_word()
user_input = ""
MAX_CHARS = 30

current_word_start_time = time.time()
normal_word_timer = enemy_ai.get_normal_word_time()
normal_word_timer_active = True

shake_screen_timer = 0
shake_enemy_timer = 0
shake_intensity_screen = 8
shake_intensity_enemy = 6

enemy_attack_timer = 0
spell_feedback_timer = 0
spell_feedback_color = WHITE

announcer_text = ""
announcer_timer = 0
announcer_duration = 90

correct_chars = 0
total_chars = 0
completed_words = 0
start_time = time.time()
enemy_kills = 0

last_difficulty_update = time.time()
difficulty_update_interval = 10

ENEMY_FRAME_X, ENEMY_FRAME_Y = 580, 330
PLAYER_FRAME_X, PLAYER_FRAME_Y = 10, 40
TEXTBOX_X, TEXTBOX_Y = 0, 445
ENEMY_HP_OFFSET_X, ENEMY_HP_OFFSET_Y = 117, 38
PLAYER_HP_OFFSET_X, PLAYER_HP_OFFSET_Y = 135, 38
SPELL_OFFSET_X, SPELL_OFFSET_Y = 240, 14
INPUT_OFFSET_X, INPUT_OFFSET_Y = 40, 60

RESTART_RECT = pg.Rect(WIDTH//2 - 140, HEIGHT//2 + 30, 120, 36)
QUIT_RECT = pg.Rect(WIDTH//2 + 20, HEIGHT//2 + 30, 120, 36)

# ML tracking variables
ml_predictions_history = []
ml_correct_predictions = 0
ml_total_predictions = 0

# ============================
# HELPERS - WITH ML INTEGRATION
# ============================
def draw_hp_bar(surface, x, y, w, h, value, max_value, is_enemy=False, is_dead=False):
    pct = max(0.0, min(1.0, value / max_value))
    
    if is_dead:
        color = DARK_GRAY
    elif pct > 0.5:
        color = GREEN
    elif pct > 0.2:
        color = YELLOW
    else:
        color = RED
    
    bg_color = (20,20,20) if not is_dead else (40, 40, 40)
    pg.draw.rect(surface, bg_color, (x, y, w, h))
    
    if not is_dead:
        inner_w = max(0, int((w - 4) * pct))
        pg.draw.rect(surface, color, (x+2, y+2, inner_w, h-4))
    
    border_color = WHITE if not is_dead else GRAY
    pg.draw.rect(surface, border_color, (x, y, w, h), 2)

def get_shake_offsets():
    sx = random.randint(-shake_intensity_screen, shake_intensity_screen) if shake_screen_timer>0 else 0
    sy = random.randint(-shake_intensity_screen, shake_intensity_screen) if shake_screen_timer>0 else 0
    return sx, sy

def get_enemy_shake_offsets():
    ex = random.randint(-shake_intensity_enemy, shake_intensity_enemy) if shake_enemy_timer>0 else 0
    ey = random.randint(-shake_intensity_enemy, shake_intensity_enemy) if shake_enemy_timer>0 else 0
    return ex, ey

def start_announcer(msg, duration=90):
    global announcer_text, announcer_timer
    announcer_text = msg
    announcer_timer = duration

def apply_damage_to_player(damage):
    global player_hp, shake_screen_timer, shake_intensity_screen, enemy_attack_timer
    player_hp = max(0, player_hp - damage)
    shake_screen_timer = 8 + int(damage * 0.8)
    global shake_intensity_screen
    shake_intensity_screen = min(30, 6 + int(damage * 0.6))
    enemy_attack_timer = 14
    start_announcer("Enemy hits you!")

def apply_damage_to_enemy(damage):
    global enemy_hp, shake_enemy_timer, shake_intensity_enemy, enemy_dead, enemy_death_timer, enemy_kills
    
    if enemy_dead:
        return
    
    enemy_hp = max(0, enemy_hp - damage)
    
    if enemy_hp <= 0 and not enemy_dead:
        enemy_dead = True
        enemy_death_timer = ENEMY_RESURRECTION_TIME
        enemy_kills += 1
        start_announcer("ENEMY DEFEATED! Resurrecting...", 120)
        
        data_collector.record_event(
            'enemy_defeated',
            kill_count=enemy_kills,
            difficulty=enemy_ai.game_difficulty
        )
    
    if not enemy_dead:
        shake_enemy_timer = 8 + int(damage * 0.4)
        shake_intensity_enemy = min(20, 4 + int(damage * 0.4))
        start_announcer("Hit!")

def get_wpm():
    elapsed_min = (time.time() - start_time) / 60
    if elapsed_min <= 0:
        return 0
    return int((correct_chars / 5) / elapsed_min)

def get_accuracy():
    return int((correct_chars / total_chars) * 100) if total_chars > 0 else 100

def update_game_difficulty():
    global last_difficulty_update
    
    current_time = time.time()
    if current_time - last_difficulty_update >= difficulty_update_interval:
        wpm = get_wpm()
        accuracy = get_accuracy()
        time_since_start = current_time - start_time
        
        current_difficulty = enemy_ai.update_game_difficulty(wpm, accuracy, time_since_start)
        
        # Use ML to adjust difficulty if trained
        if ml_predictor.trained:
            player_stats = get_player_stats_for_ml()
            if player_stats:
                ml_difficulty = ml_predictor.get_optimal_difficulty(player_stats)
                # Blend ML suggestion with AI difficulty (70% ML, 30% AI)
                blended_difficulty = (ml_difficulty * 0.7) + (current_difficulty * 0.3)
                enemy_ai.game_difficulty = min(10, max(1, blended_difficulty))
                current_difficulty = enemy_ai.game_difficulty
        
        data_collector.record_event(
            'difficulty_update',
            difficulty=current_difficulty,
            wpm=wpm,
            accuracy=accuracy,
            time_played=time_since_start,
            enemy_kills=enemy_kills
        )
        
        last_difficulty_update = current_time
        
        if current_difficulty > 5:
            start_announcer(f"Difficulty Increased! (Level {int(current_difficulty)})", 90)
        
        return current_difficulty
    
    return enemy_ai.game_difficulty

def adjust_words_per_turn():
    global WORDS_PER_TURN
    
    difficulty = enemy_ai.game_difficulty
    
    if difficulty > 8:
        return 2
    elif difficulty > 6:
        return random.choice([2, 3])
    elif difficulty > 4:
        return 3
    else:
        return 4

def get_player_stats_for_ml():
    """Extract player statistics for ML prediction"""
    recent_attempts = [e for e in data_collector.session_data 
                      if e['type'] == 'word_attempt']
    
    if len(recent_attempts) < 4:
        return None
    
    last_four = recent_attempts[-4:]
    prev_three = last_four[-4:-1]
    current = last_four[-1]
    
    prev_correct = 1 if prev_three[-1]['data']['correct'] else 0
    prev_time = prev_three[-1]['data']['time_taken']
    prev_difficulty = prev_three[-1]['data']['word_difficulty']
    
    current_length = len(current['data']['word'])
    current_difficulty = current['data']['word_difficulty']
    
    correct_in_last_three = sum(1 for a in prev_three if a['data']['correct'])
    rolling_accuracy = correct_in_last_three / 3.0
    
    total_time = sum(a['data']['time_taken'] for a in prev_three)
    if total_time > 0:
        total_chars = sum(len(a['data']['word']) for a in prev_three)
        avg_wpm = (total_chars / 5) / (total_time / 60)
    else:
        avg_wpm = 0
    
    return [
        prev_correct, prev_time, prev_difficulty,
        current_length, current_difficulty,
        rolling_accuracy, avg_wpm
    ]

def make_ml_prediction():
    """Make ML prediction for next word success"""
    global ml_total_predictions, ml_correct_predictions
    
    if not ml_predictor.trained:
        return 0.5, 5
    
    player_stats = get_player_stats_for_ml()
    if not player_stats:
        return 0.5, 5
    
    success_prob = ml_predictor.predict_success_probability(player_stats)
    optimal_difficulty = ml_predictor.get_optimal_difficulty(player_stats)
    
    # Record prediction for accuracy tracking
    ml_total_predictions += 1
    ml_predictions_history.append({
        'probability': success_prob,
        'timestamp': time.time() - start_time
    })
    
    if len(ml_predictions_history) > 50:
        ml_predictions_history.pop(0)
    
    return success_prob, optimal_difficulty

def handle_counter_success():
    global game_phase, counter_success, user_input
    counter_success = True
    user_input = ""
    
    base_damage = 25
    difficulty_bonus = enemy_ai.game_difficulty * 2
    total_damage = base_damage + difficulty_bonus
    
    apply_damage_to_enemy(total_damage)
    start_announcer("SUCCESSFUL COUNTER!", 60)
    
    data_collector.record_event(
        'counter_success',
        word=enemy_attack_word,
        difficulty=enemy_ai.game_difficulty,
        damage_dealt=total_damage
    )
    
    pg.time.set_timer(pg.USEREVENT, 1000)

def handle_counter_failure():
    global game_phase, counter_failed, user_input
    counter_failed = True
    user_input = ""
    
    base_damage = 20
    difficulty_bonus = enemy_ai.game_difficulty * 1.5
    total_damage = base_damage + difficulty_bonus
    
    apply_damage_to_player(total_damage)
    start_announcer("COUNTER FAILED!", 60)
    
    data_collector.record_event(
        'counter_failure',
        word=enemy_attack_word,
        difficulty=enemy_ai.game_difficulty,
        damage_taken=total_damage
    )
    
    pg.time.set_timer(pg.USEREVENT, 1000)

def start_enemy_attack_phase():
    global game_phase, enemy_attack_word, counter_timer, words_typed_this_turn
    
    if enemy_dead:
        return
    
    game_phase = ENEMY_ATTACK_PHASE
    enemy_attack_word = enemy_ai.choose_counter_word()
    
    counter_time = enemy_ai.get_counter_time_based_on_difficulty(enemy_attack_word)
    counter_timer = counter_time
    
    words_typed_this_turn = 0
    
    difficulty = enemy_ai.game_difficulty
    if difficulty > 7:
        start_announcer("CRITICAL ATTACK! Type fast!", 120)
    elif difficulty > 5:
        start_announcer("ENEMY ATTACK! Quick counter!", 120)
    else:
        start_announcer("Enemy attack! Counter quickly!", 120)
    
    data_collector.record_event(
        'enemy_attack',
        word=enemy_attack_word,
        difficulty=enemy_ai.game_difficulty,
        word_difficulty=word_lib.word_difficulty.get(enemy_attack_word, 5),
        counter_time=counter_time
    )

def handle_word_submission(typed_word):
    global current_word, user_input, words_typed_this_turn, current_word_start_time
    global correct_chars, total_chars, completed_words, spell_feedback_timer, spell_feedback_color
    global normal_word_timer, normal_word_timer_active, WORDS_PER_TURN
    global ml_correct_predictions
    
    typing_time = time.time() - current_word_start_time
    correct = typed_word == current_word
    
    enemy_ai.record_typing_event(
        word=current_word,
        typed_word=typed_word,
        correct=correct,
        time_taken=typing_time
    )
    
    data_collector.record_event(
        'word_attempt',
        word=current_word,
        typed=typed_word,
        correct=correct,
        time_taken=typing_time,
        wrong_letters=[],
        word_difficulty=word_lib.word_difficulty.get(current_word, 5),
        game_difficulty=enemy_ai.game_difficulty
    )
    
    total_chars += len(typed_word)
    difficulty = enemy_ai.game_difficulty
    
    # Make ML prediction for next word
    ml_success_prob, ml_optimal_diff = make_ml_prediction()
    
    # Check if ML prediction was correct (if we made one)
    if ml_total_predictions > 0 and len(ml_predictions_history) > 0:
        last_pred = ml_predictions_history[-1]
        predicted_success = last_pred['probability'] > 0.5
        if predicted_success == correct:
            ml_correct_predictions += 1
    
    if correct:
        correct_chars += len(current_word)
        completed_words += 1
        
        base_damage = 15
        word_difficulty_bonus = word_lib.word_difficulty.get(current_word, 5)
        difficulty_bonus = difficulty * 1.5
        
        total_damage = base_damage + word_difficulty_bonus + difficulty_bonus
        apply_damage_to_enemy(total_damage)
        spell_feedback_timer = 30
        spell_feedback_color = GREEN
        
        data_collector.record_event(
            'successful_hit',
            damage=total_damage,
            word_difficulty=word_difficulty_bonus,
            game_difficulty=difficulty,
            ml_prediction=ml_success_prob,
            ml_optimal_difficulty=ml_optimal_diff
        )
    else:
        base_damage = 10
        word_difficulty_bonus = word_lib.word_difficulty.get(current_word, 5) // 2
        difficulty_bonus = difficulty * 1.2
        
        total_damage = base_damage + word_difficulty_bonus + difficulty_bonus
        apply_damage_to_player(total_damage)
        spell_feedback_timer = 30
        spell_feedback_color = RED
        
        data_collector.record_event(
            'failed_hit',
            damage_taken=total_damage,
            word_difficulty=word_difficulty_bonus,
            game_difficulty=difficulty,
            ml_prediction=ml_success_prob,
            ml_optimal_difficulty=ml_optimal_diff
        )
    
    words_typed_this_turn += 1
    
    normal_word_timer = enemy_ai.get_normal_word_time()
    normal_word_timer_active = True
    current_word_start_time = time.time()
    
    WORDS_PER_TURN = adjust_words_per_turn()
    
    if not enemy_dead:
        if words_typed_this_turn >= WORDS_PER_TURN:
            start_enemy_attack_phase()
        elif enemy_ai.should_increase_attack_frequency() and words_typed_this_turn >= 2:
            start_enemy_attack_phase()
        else:
            # Use ML to influence word selection
            if ml_predictor.trained:
                # Adjust difficulty based on ML prediction
                if ml_success_prob > 0.8:
                    # High success probability - increase difficulty
                    enemy_ai.game_difficulty = min(10, enemy_ai.game_difficulty + 0.3)
                elif ml_success_prob < 0.4:
                    # Low success probability - decrease difficulty
                    enemy_ai.game_difficulty = max(1, enemy_ai.game_difficulty - 0.3)
            
            current_word = enemy_ai.choose_challenging_word()
            current_word_start_time = time.time()
    else:
        current_word = enemy_ai.choose_challenging_word()
        current_word_start_time = time.time()
    
    user_input = ""

# ============================
# MAIN LOOP
# ============================
running = True
while running:
    dt = clock.get_time() / 1000.0
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            data_collector.save_session()
            running = False
        
        if event.type == pg.USEREVENT:
            pg.time.set_timer(pg.USEREVENT, 0)
            if game_phase == ENEMY_ATTACK_PHASE:
                game_phase = NORMAL_PHASE
                counter_success = False
                counter_failed = False
                current_word = enemy_ai.choose_challenging_word()
                current_word_start_time = time.time()
                user_input = ""
                normal_word_timer = enemy_ai.get_normal_word_time()
                normal_word_timer_active = True

        if game_over:
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                mx,my = event.pos
                if RESTART_RECT.collidepoint(mx,my):
                    game_over = False
                    game_over_fade = 0.0
                    enemy_hp = enemy_hp_max
                    player_hp = player_hp_max
                    disp_enemy_hp = float(enemy_hp)
                    disp_player_hp = float(player_hp)
                    current_word = enemy_ai.choose_challenging_word()
                    user_input = ""
                    correct_chars = total_chars = completed_words = 0
                    enemy_kills = 0
                    words_typed_this_turn = 0
                    game_phase = NORMAL_PHASE
                    current_word_start_time = time.time()
                    normal_word_timer = enemy_ai.get_normal_word_time()
                    normal_word_timer_active = True
                    start_time = time.time()
                    last_difficulty_update = time.time()
                    enemy_dead = False
                    enemy_death_timer = 0
                    enemy_resurrection_progress = 0
                    enemy_ai = PatternExploiterAI(word_lib)
                    ml_predictions_history = []
                    ml_correct_predictions = 0
                    ml_total_predictions = 0
                if QUIT_RECT.collidepoint(mx,my):
                    data_collector.save_session()
                    running = False
            if event.type == pg.KEYDOWN:
                if event.key in (pg.K_SPACE, pg.K_r):
                    game_over = False
                    game_over_fade = 0.0
                    enemy_hp = enemy_hp_max
                    player_hp = player_hp_max
                    disp_enemy_hp = float(enemy_hp)
                    disp_player_hp = float(player_hp)
                    current_word = enemy_ai.choose_challenging_word()
                    user_input = ""
                    correct_chars = total_chars = completed_words = 0
                    enemy_kills = 0
                    words_typed_this_turn = 0
                    game_phase = NORMAL_PHASE
                    current_word_start_time = time.time()
                    normal_word_timer = enemy_ai.get_normal_word_time()
                    normal_word_timer_active = True
                    start_time = time.time()
                    last_difficulty_update = time.time()
                    enemy_dead = False
                    enemy_death_timer = 0
                    enemy_resurrection_progress = 0
                    enemy_ai = PatternExploiterAI(word_lib)
                    ml_predictions_history = []
                    ml_correct_predictions = 0
                    ml_total_predictions = 0
            continue

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_BACKSPACE:
                user_input = user_input[:-1]
            elif event.key == pg.K_RETURN:
                if game_phase == NORMAL_PHASE:
                    handle_word_submission(user_input.strip())
                elif game_phase == ENEMY_ATTACK_PHASE:
                    typed = user_input.strip()
                    if typed == enemy_attack_word:
                        correct_chars += len(enemy_attack_word)
                        completed_words += 1
                        handle_counter_success()
                    else:
                        handle_counter_failure()
            elif event.key == pg.K_CAPSLOCK:
                pass
            else:
                if len(user_input) < MAX_CHARS:
                    if event.unicode:
                        user_input += event.unicode
                        
                        if game_phase == ENEMY_ATTACK_PHASE and not counter_success and not counter_failed:
                            counter_timer = enemy_ai.get_counter_time_based_on_difficulty(enemy_attack_word)
                        
                        if game_phase == NORMAL_PHASE:
                            normal_word_timer = enemy_ai.get_normal_word_time()
                            normal_word_timer_active = True

    current_difficulty = update_game_difficulty()
    
    if enemy_dead:
        enemy_death_timer -= 1
        enemy_resurrection_progress = 1.0 - (enemy_death_timer / ENEMY_RESURRECTION_TIME)
        
        if enemy_death_timer <= 0:
            enemy_dead = False
            enemy_hp = enemy_hp_max
            disp_enemy_hp = float(enemy_hp)
            enemy_resurrection_progress = 0
            start_announcer("ENEMY RESURRECTED!", 90)
            
            enemy_ai.game_difficulty = min(10, enemy_ai.game_difficulty + 0.5)
            
            data_collector.record_event(
                'enemy_resurrected',
                kill_count=enemy_kills,
                new_difficulty=enemy_ai.game_difficulty
            )
    
    if game_phase == NORMAL_PHASE and normal_word_timer_active and not enemy_dead:
        normal_word_timer -= dt
        if normal_word_timer <= 0:
            normal_word_timer_active = False
            handle_word_submission("")
    
    if game_phase == ENEMY_ATTACK_PHASE and not counter_success and not counter_failed and not enemy_dead:
        counter_timer -= dt
        if counter_timer <= 0:
            handle_counter_failure()
    
    if shake_screen_timer > 0:
        shake_screen_timer -= 1
    else:
        shake_intensity_screen = 8

    if shake_enemy_timer > 0:
        shake_enemy_timer -= 1
    else:
        shake_intensity_enemy = 6

    if enemy_attack_timer > 0:
        enemy_attack_timer -= 1

    if spell_feedback_timer > 0:
        spell_feedback_timer = max(0, spell_feedback_timer - 1)
        if spell_feedback_timer == 0:
            spell_feedback_color = WHITE

    if announcer_timer > 0:
        announcer_timer -= 1
        if announcer_timer == 0 and game_over:
            pass

    lerp_speed = 0.12
    disp_enemy_hp += (enemy_hp - disp_enemy_hp) * lerp_speed
    disp_player_hp += (player_hp - disp_player_hp) * lerp_speed

    if player_hp <= 0 and not game_over:
        game_over = True
        game_over_fade = 0.0
        
        # Calculate live ML accuracy
        live_ml_accuracy = 0
        if ml_total_predictions > 0:
            live_ml_accuracy = ml_correct_predictions / ml_total_predictions
        
        data_collector.record_event(
            'game_over',
            final_difficulty=enemy_ai.game_difficulty,
            total_words=completed_words,
            final_wpm=get_wpm(),
            final_accuracy=get_accuracy(),
            play_time=time.time() - start_time,
            enemy_kills=enemy_kills,
            live_ml_accuracy=live_ml_accuracy,
            ml_predictions_made=ml_total_predictions,
            ml_correct_predictions=ml_correct_predictions
        )
        data_collector.save_session()

    sx, sy = get_shake_offsets()
    ex, ey = get_enemy_shake_offsets()

    screen.blit(battle_bg, (sx, sy))

    lunge_offset = 0
    if enemy_attack_timer > 0 and not enemy_dead:
        t = enemy_attack_timer / 14.0
        lunge_offset = int(12 * math.sin((1-t) * math.pi))
    
    enemy_sprite_pos = (670 - 90 + ex + sx + lunge_offset, 100 + 20 + ey + sy)
    
    if enemy_dead:
        screen.blit(enemy_sprite_grayscale, enemy_sprite_pos)
        
        death_overlay = pg.Surface((enemy_sprite_img.get_width(), enemy_sprite_img.get_height()), pg.SRCALPHA)
        death_alpha = 150 + int(50 * (1.0 - enemy_resurrection_progress))
        death_overlay.fill((0, 0, 0, death_alpha))
        screen.blit(death_overlay, enemy_sprite_pos)
        
        if enemy_resurrection_progress > 0:
            res_width = 100
            res_height = 8
            res_x = enemy_sprite_pos[0] + enemy_sprite_img.get_width() // 2 - res_width // 2
            res_y = enemy_sprite_pos[1] + enemy_sprite_img.get_height() + 5
            
            pg.draw.rect(screen, (40, 40, 40), (res_x, res_y, res_width, res_height))
            progress_width = int(res_width * enemy_resurrection_progress)
            pg.draw.rect(screen, (180, 80, 220), (res_x, res_y, progress_width, res_height))
            pg.draw.rect(screen, WHITE, (res_x, res_y, res_width, res_height), 1)
            
            res_text = font_small.render("RESURRECTING", True, PURPLE)
            screen.blit(res_text, (res_x + res_width // 2 - res_text.get_width() // 2, res_y - 15))
    else:
        screen.blit(enemy_sprite_img, enemy_sprite_pos)
    
    enemy_frame_pos = (ENEMY_FRAME_X + sx + ex, ENEMY_FRAME_Y + sy + ey)
    screen.blit(enemy_frame, enemy_frame_pos)
    
    draw_hp_bar(screen,
                enemy_frame_pos[0] + ENEMY_HP_OFFSET_X,
                enemy_frame_pos[1] + ENEMY_HP_OFFSET_Y,
                98, 10,
                disp_enemy_hp, enemy_hp_max,
                is_enemy=True,
                is_dead=enemy_dead)

    player_sprite_pos = (180 + 30 + sx, 160 - 10 + sy)
    screen.blit(player_sprite_img, player_sprite_pos)
    player_frame_pos = (PLAYER_FRAME_X + sx, PLAYER_FRAME_Y + sy)
    screen.blit(player_frame, player_frame_pos)
    draw_hp_bar(screen,
                player_frame_pos[0] + PLAYER_HP_OFFSET_X,
                player_frame_pos[1] + PLAYER_HP_OFFSET_Y,
                98, 10,
                disp_player_hp, player_hp_max)

    textbox_pos = (TEXTBOX_X + sx, TEXTBOX_Y + sy)
    screen.blit(textbox_bg, textbox_pos)

    if game_phase == NORMAL_PHASE:
        spell_color = spell_feedback_color if spell_feedback_timer > 0 else WHITE
        
        word_to_display = current_word
        word_surface = font_big.render(word_to_display, True, spell_color)
        max_word_width = 600
        
        if word_surface.get_width() > max_word_width:
            word_to_display = word_to_display[:15] + "..."
        
        spell_surf_shadow = font_big.render(word_to_display, True, BLACK)
        spell_surf_main = font_big.render(word_to_display, True, spell_color)
        screen.blit(spell_surf_shadow, (textbox_pos[0] + SPELL_OFFSET_X + 2, textbox_pos[1] + SPELL_OFFSET_Y + 2))
        screen.blit(spell_surf_main, (textbox_pos[0] + SPELL_OFFSET_X, textbox_pos[1] + SPELL_OFFSET_Y))
        
        words_for_next_attack = WORDS_PER_TURN - words_typed_this_turn if not enemy_dead else "ENEMY DEAD"
        turn_counter_text = f"Next attack in: {words_for_next_attack}"
        turn_counter_surf = font_small.render(turn_counter_text, True, PURPLE if not enemy_dead else GRAY)
        screen.blit(turn_counter_surf, (textbox_pos[0] + 20, textbox_pos[1] + 100))
        
        if not enemy_dead:
            base_time = enemy_ai.get_normal_word_time()
            timer_text = f"Time: {normal_word_timer:.1f}s"
            time_percentage = normal_word_timer / base_time if base_time > 0 else 0
            if time_percentage > 0.6:
                timer_color = GREEN
            elif time_percentage > 0.3:
                timer_color = YELLOW
            else:
                timer_color = RED
        else:
            timer_text = "ENEMY DEAD"
            timer_color = GRAY
            
        timer_surf = font_small.render(timer_text, True, timer_color)
        screen.blit(timer_surf, (textbox_pos[0] + 250, textbox_pos[1] + 100))
        
        difficulty_text = f"Difficulty: {int(current_difficulty)}/10"
        diff_surf = font_small.render(difficulty_text, True, YELLOW)
        screen.blit(diff_surf, (textbox_pos[0] + 400, textbox_pos[1] + 100))
        
        kills_text = f"Enemies defeated: {enemy_kills}"
        kills_surf = font_small.render(kills_text, True, CYAN)
        screen.blit(kills_surf, (textbox_pos[0] + 550, textbox_pos[1] + 100))
        
    elif game_phase == ENEMY_ATTACK_PHASE:
        warning_alpha = 100 + int(enemy_ai.game_difficulty * 15)
        warning_alpha = min(200, warning_alpha)
        warning_bg = pg.Surface((WIDTH, 140), pg.SRCALPHA)
        warning_bg.fill((180, 80, 220, warning_alpha))
        screen.blit(warning_bg, (textbox_pos[0], textbox_pos[1]))
        
        counter_text_str = "COUNTER ATTACK!"
        if enemy_ai.game_difficulty > 8:
            counter_text_str = "CRITICAL COUNTER ATTACK!"
        elif enemy_ai.game_difficulty > 6:
            counter_text_str = "DANGEROUS COUNTER ATTACK!"
        
        counter_text = font_med.render(counter_text_str, True, CYAN)
        screen.blit(counter_text, (textbox_pos[0] + WIDTH//2 - counter_text.get_width()//2, textbox_pos[1] + 10))
        
        attack_word_to_display = enemy_attack_word
        attack_word_surface = font_big.render(attack_word_to_display, True, RED)
        max_attack_width = 800
        
        if attack_word_surface.get_width() > max_attack_width:
            attack_word_to_display = attack_word_to_display[:20] + "..."
        
        attack_word_shadow = font_big.render(attack_word_to_display, True, BLACK)
        attack_word_main = font_big.render(attack_word_to_display, True, RED)
        screen.blit(attack_word_shadow, (textbox_pos[0] + WIDTH//2 - attack_word_shadow.get_width()//2 + 2, 
                                        textbox_pos[1] + 50 + 2))
        screen.blit(attack_word_main, (textbox_pos[0] + WIDTH//2 - attack_word_main.get_width()//2, 
                                      textbox_pos[1] + 50))
        
        timer_text = f"Time: {counter_timer:.1f}s"
        word_difficulty = word_lib.word_difficulty.get(enemy_attack_word, 5)
        
        base_time = enemy_ai.get_counter_time_based_on_difficulty(enemy_attack_word)
        time_percentage = counter_timer / base_time if base_time > 0 else 0
        if time_percentage > 0.6:
            timer_color = GREEN
        elif time_percentage > 0.3:
            timer_color = YELLOW
        else:
            timer_color = RED
            
        timer_surf = font_med.render(timer_text, True, timer_color)
        screen.blit(timer_surf, (textbox_pos[0] + WIDTH//2 - timer_surf.get_width()//2, textbox_pos[1] + 100))
        
        diff_text = f"Word Difficulty: {word_difficulty}/10  |  Game Difficulty: {int(enemy_ai.game_difficulty)}/10"
        diff_surf = font_small.render(diff_text, True, WHITE)
        screen.blit(diff_surf, (textbox_pos[0] + WIDTH//2 - diff_surf.get_width()//2, textbox_pos[1] + 130))
        
        if counter_success:
            result_text = font_med.render("SUCCESSFUL COUNTER!", True, GREEN)
            screen.blit(result_text, (textbox_pos[0] + WIDTH//2 - result_text.get_width()//2, textbox_pos[1] + 150))
        elif counter_failed:
            result_text = font_med.render("COUNTER FAILED!", True, RED)
            screen.blit(result_text, (textbox_pos[0] + WIDTH//2 - result_text.get_width()//2, textbox_pos[1] + 150))

    input_to_display = user_input
    input_surface = font_med.render(input_to_display, True, BLACK)
    max_input_width = 800
    
    if input_surface.get_width() > max_input_width:
        input_to_display = "..." + user_input[-30:]
    
    input_shadow = font_med.render(input_to_display, True, WHITE)
    input_main = font_med.render(input_to_display, True, BLACK)
    screen.blit(input_shadow, (textbox_pos[0] + INPUT_OFFSET_X + 3, textbox_pos[1] + INPUT_OFFSET_Y + 3))
    screen.blit(input_main, (textbox_pos[0] + INPUT_OFFSET_X, textbox_pos[1] + INPUT_OFFSET_Y))

    if announcer_timer > 0:
        ann_w, ann_h = 720, 50
        ann_x = (WIDTH - ann_w)//2 + sx
        ann_y = textbox_pos[1] - ann_h - 8
        ann_surf = pg.Surface((ann_w, ann_h), pg.SRCALPHA)
        ann_surf.fill((10,10,10,220))
        pg.draw.rect(ann_surf, WHITE, (0,0,ann_w,ann_h), 2)
        ann_text = announcer_text if announcer_text else ""
        ann_render = font_med.render(ann_text, True, WHITE)
        ann_surf.blit(ann_render, (12, 12))
        screen.blit(ann_surf, (ann_x, ann_y))

    accuracy = get_accuracy()
    wpm = get_wpm()
    stats_text = f"Correct: {completed_words}  WPM: {wpm}   ACC: {accuracy}%"
    stats_shadow = font_small.render(stats_text, True, BLACK)
    stats_main = font_small.render(stats_text, True, WHITE)
    screen.blit(stats_shadow, (11 + sx, 11 + sy))
    screen.blit(stats_main, (10 + sx, 10 + sy))

    # ML MODEL DISPLAY
    if ml_predictor.trained:
        # Show ML model accuracy from training
        ml_acc_text = f"ML Acc: {ml_predictor.metrics['accuracy']:.1%}"
        ml_acc_surf = font_small.render(ml_acc_text, True, ML_COLOR)
        screen.blit(ml_acc_surf, (WIDTH - 150 + sx, 10 + sy))
        
        # Show live ML prediction accuracy
        live_ml_acc = 0
        if ml_total_predictions > 0:
            live_ml_acc = ml_correct_predictions / ml_total_predictions
            live_acc_text = f"Live: {live_ml_acc:.1%}"
            live_acc_surf = font_small.render(live_acc_text, True, GREEN if live_ml_acc > 0.7 else YELLOW if live_ml_acc > 0.5 else RED)
            screen.blit(live_acc_surf, (WIDTH - 150 + sx, 30 + sy))
        
        # Show current ML prediction if we have recent stats
        player_stats = get_player_stats_for_ml()
        if player_stats:
            success_prob = ml_predictor.predict_success_probability(player_stats)
            pred_text = f"ML Pred: {success_prob:.0%}"
            pred_color = GREEN if success_prob > 0.7 else YELLOW if success_prob > 0.4 else RED
            pred_surf = font_small.render(pred_text, True, pred_color)
            screen.blit(pred_surf, (WIDTH - 150 + sx, 50 + sy))
            
            # Show ML-suggested difficulty
            ml_diff = ml_predictor.get_optimal_difficulty(player_stats)
            diff_text = f"ML Diff: {ml_diff}"
            diff_surf = font_small.render(diff_text, True, CYAN)
            screen.blit(diff_surf, (WIDTH - 150 + sx, 70 + sy))
            
            # Show most important feature
            if ml_predictor.metrics['feature_importance']:
                top_feature = max(ml_predictor.metrics['feature_importance'].items(), key=lambda x: abs(x[1]))
                feature_text = f"Top: {top_feature[0]}"
                feature_surf = font_small.render(feature_text, True, WHITE)
                screen.blit(feature_surf, (WIDTH - 150 + sx, 90 + sy))
    else:
        ml_text = "ML: Not Trained"
        ml_surf = font_small.render(ml_text, True, GRAY)
        screen.blit(ml_surf, (WIDTH - 150 + sx, 10 + sy))

    ai_info_text = f"AI: {enemy_ai.personality.upper()}"
    ai_info_surf = font_small.render(ai_info_text, True, CYAN)
    screen.blit(ai_info_surf, (WIDTH - 150 + sx, 110 + sy))
    
    if game_phase == NORMAL_PHASE:
        diff = word_lib.word_difficulty.get(current_word, 5)
        diff_text = f"Word Difficulty: {diff}/10"
        diff_color = GREEN if diff <= 3 else YELLOW if diff <= 6 else RED
        diff_surf = font_small.render(diff_text, True, diff_color)
        screen.blit(diff_surf, (WIDTH - 180 + sx, 130 + sy))
        
        if not enemy_dead:
            attack_freq_text = f"Attack freq: {WORDS_PER_TURN}"
        else:
            attack_freq_text = "ENEMY DEAD"
        attack_surf = font_small.render(attack_freq_text, True, WHITE if not enemy_dead else GRAY)
        screen.blit(attack_surf, (WIDTH - 180 + sx, 150 + sy))
    
    if game_phase == ENEMY_ATTACK_PHASE:
        base_time = enemy_ai.get_counter_time_based_on_difficulty(enemy_attack_word)
        time_info = f"Base time: {base_time:.1f}s"
        time_info_surf = font_small.render(time_info, True, YELLOW)
        screen.blit(time_info_surf, (WIDTH - 180 + sx, 170 + sy))
        
        normal_time_info = f"Normal word time: {enemy_ai.get_normal_word_time():.1f}s"
        normal_time_surf = font_small.render(normal_time_info, True, CYAN)
        screen.blit(normal_time_surf, (WIDTH - 200 + sx, 190 + sx))

    if shake_screen_timer > 0:
        alpha = min(160, 40 + (shake_screen_timer * 8))
        flash = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
        flash.fill((255,255,255, alpha))
        screen.blit(flash, (0,0))

    if game_over:
        game_over_fade = min(1.0, game_over_fade + 0.02)
        overlay = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
        overlay.fill((0,0,0, int(180 * game_over_fade)))
        screen.blit(overlay, (0,0))
        
        msg = "YOU LOST"
        msg_shadow = font_big.render(msg, True, BLACK)
        msg_main = font_big.render(msg, True, (220, 30, 30))
        mx = WIDTH//2 - msg_main.get_width()//2
        my = HEIGHT//2 - msg_main.get_height()//2 - 40
        screen.blit(msg_shadow, (mx+3, my+3))
        screen.blit(msg_main, (mx, my))
        
        final_stats = f"Final Difficulty: {int(enemy_ai.game_difficulty)}/10  |  Words: {completed_words}  |  WPM: {get_wpm()}"
        final_stats_surf = font_med.render(final_stats, True, WHITE)
        screen.blit(final_stats_surf, (WIDTH//2 - final_stats_surf.get_width()//2, my + 60))
        
        # Show ML performance in game over screen
        if ml_predictor.trained and ml_total_predictions > 0:
            live_ml_acc = ml_correct_predictions / ml_total_predictions
            ml_perf_text = f"ML Accuracy: {ml_predictor.metrics['accuracy']:.1%} (Train) | {live_ml_acc:.1%} (Live)"
            ml_perf_surf = font_med.render(ml_perf_text, True, ML_COLOR)
            screen.blit(ml_perf_surf, (WIDTH//2 - ml_perf_surf.get_width()//2, my + 150))
        
        kills_final = f"Enemies defeated: {enemy_kills}"
        kills_final_surf = font_med.render(kills_final, True, CYAN)
        screen.blit(kills_final_surf, (WIDTH//2 - kills_final_surf.get_width()//2, my + 120))
        
        if game_over_fade >= 0.6:
            pg.draw.rect(screen, (30, 120, 40), RESTART_RECT)
            pg.draw.rect(screen, (120, 30, 30), QUIT_RECT)
            rtxt = font_small.render("RESTART (Space/R)", True, WHITE)
            qtxt = font_small.render("QUIT", True, WHITE)
            screen.blit(rtxt, (RESTART_RECT.x + 8, RESTART_RECT.y + 8))
            screen.blit(qtxt, (QUIT_RECT.x + 36, QUIT_RECT.y + 8))

    pg.display.flip()
    clock.tick(FPS)

pg.quit()
from inspect import cleandoc
import torch
from PIL import Image, PngImagePlugin, ImageDraw, ImageFont

import numpy as np
import folder_paths
import datetime
import os
import json
import random
import time
import re


class AsunaroWildCard:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pick_up_num": ("INT", {"default": 1, "min": 0, "step": 1}),
                "wildcard": ("STRING", {
                    "multiline": True,
                }),
                "delimiter": ("STRING", {"default": ",", "multiline": False}),

                "seed": ("INT:seed", {}),

            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "asunaro_wildcard"

    def asunaro_wildcard(self, wildcard, delimiter, pick_up_num, seed):
        random.seed(seed)
        w_list = wildcard.replace('\n', '').split(delimiter)

        if pick_up_num > len(w_list):
            pick_up_num = len(w_list)
            print (f"Error: pick_up_num is too large. The number of wildcards is {len(w_list)}",)
        elif pick_up_num < 1:
            return ("",)
        result = ""

        result += ",".join(random.sample(w_list, pick_up_num))
        if not result:
            return ("",)
        return (result,)

class AsunaroIfSame:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("STRING",),
                "compare": ("STRING", {
                    "multiline": False,
                }),
                "if_true": ("INT", {"default": 1, "min": 0, "step": 1}),
                "if_false": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "asunaro_if_int"

    def asunaro_if_int(self, target, compare, if_true, if_false):
        if compare == target:
            return (if_true,)
        else:
            return (if_false,)

class AsunaroIfContain:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "haystack": ("STRING",),
                "needle": ("STRING", {
                    "multiline": False,
                }),
                "if_true": ("INT", {"default": 1, "min": 0, "step": 1}),
                "if_false": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "asunaro_if_contain"

    def asunaro_if_contain(self, haystack, needle, if_true, if_false):
        if needle in haystack:
            return (if_true,)
        else:
            return (if_false,)

class AsunaroIfBiggerThanZero:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("INT",),
                "input": ("STRING", ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "asunaro_if_bigger_than_zero"

    def asunaro_if_bigger_than_zero(self, target, input):
        if target > 0:
            return (input,)
        else:
            return ("",)



class AsunaroAnd:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("INT",{"default": 1}),
                "input2": ("INT",{"default": 1}),
                "input3": ("INT",{"default": 1}),
                "input4": ("INT",{"default": 1}),
                "input5": ("INT",{"default": 1}),
                "if_true": ("INT", {"default": 1, "min": 0, "step": 1}),
                "if_false": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "asunaro_and"

    def asunaro_and(self, input1, input2, input3, input4, input5, if_true, if_false):
        if input1 > 0 and input2 > 0 and input3 > 0 and input4 > 0 and input5 > 0:
            return (if_true,)
        return (if_false,)


class AsunaroOr:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input1": ("INT",{"default": 0}),
                "input2": ("INT",{"default": 0}),
                "input3": ("INT",{"default": 0}),
                "input4": ("INT",{"default": 0}),
                "input5": ("INT",{"default": 0}),
                "if_true": ("INT", {"default": 1, "min": 0, "step": 1}),
                "if_false": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "asunaro_or"

    def asunaro_or(self, input1, input2, input3, input4, input5, if_true, if_false):
        if input1 > 0 or input2 > 0 or input3 > 0 or input4 > 0 or input5 > 0:
            return (if_true,)
        return (if_false,)



class AsunaroIntToStr:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 1, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "asunaro_int_to_str"

    def asunaro_int_to_str(self, int):
        return (str(int),)

class AsunaroTextConcatenator:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "delimiter": ("STRING", {"default": ",", "multiline": False}),
            },
            "optional": {
                "text1": ("STRING", ),
                "text2": ("STRING", ),
                "text3": ("STRING", ),
                "text4": ("STRING", ),
                "text5": ("STRING", ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "asunaro_text_concatenator"

    def asunaro_text_concatenator(self, text1, text2, text3, text4, text5, delimiter):
        texts = [text1, text2, text3, text4, text5]
        texts = delimiter.join(texts)
        print(texts)

        return (texts,)

class AsunaroPromptStripper:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", ),
                "strip_words": ("STRING", ),
                "add_words": ("STRING", ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "asunaro_prompt_stripper"

    def asunaro_prompt_stripper(self, prompt, strip_words, add_words):
        prompt_array = [s.strip() for s in re.split(r"\s*,+\s*", prompt.strip()) if s.strip()]
        strip_words_array = [s.strip() for s in re.split(r"\s*,+\s*", strip_words.strip()) if s.strip()]
        add_words_array = [s.strip() for s in re.split(r"\s*,+\s*", add_words.strip()) if s.strip()]

        filtered_list = [item for item in prompt_array if item not in strip_words_array]

        result = ", ".join(filtered_list + add_words_array)

        return (result,)




class AsunaroRandomDice:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min": ("INT", {"default": 1, "min": 0, "step": 1}),
                "max": ("INT", {"default": 1, "min": 0, "step": 1}),
                "seed": ("INT:seed", {}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "asunaro_random_dice"

    def asunaro_random_dice(self, min, max, seed):
        random.seed(seed)
        result = random.randint(min, max)
        return (result,)


class AsunaroAutomaticSexPrompter:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sex_mode": (["sex", "fellatio"], {"default": "sex"}),
                "insertion": (["deep", "normal", "imminent"], {"default": "normal"}),
                "cum": (["none", "normal", "excessive"], {"default": "none"}),
                "panty": (["on", "aside", "off", "random"],),
                "bra": (["on", "aside", "off", "random"],),
                "topless": ("BOOLEAN", ),
                "bottomless": ("BOOLEAN", ),
                "face": (['random', "use_face_string"], {"default": ""}),
                "face_string": ('STRING', {"default": ""}),
                "angle": (['random', "use_angle_string"], {"default": "random"}),
                "angle_string": ('STRING', {"default": ""}),
                "lighting": (['random', "use_lighting_string"], {"default": "random"}),
                "lighting_string": ('STRING', {"default": ""}),
                "focus": (["face", "pussy", "none", "random"], {"default": "none"}),
                "min_number_of_penis": ("INT", {"default": 1}),
                "max_number_of_penis": ("INT", {"default": 2}),
                "seed": ("INT:seed", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "asunaro_auto_sex"

    def asunaro_auto_sex(self, sex_mode, insertion, cum, panty, bra, topless, bottomless, face, face_string, angle, angle_string, lighting, lighting_string, focus, min_number_of_penis, max_number_of_penis, seed):
        random.seed(seed)
        positive_prompt = []
        negative_prompt = []
        face_option = ["ahegao", "frown", "ahegao", "frown", "disgust", "angry, ahegao", "angry", "surprized", "scared", "disgust"]
        angle_option = ["ceiling", "character focus", "cinematic angle", "close to viewer", "dutch angle", "dynamic angle", "feet out of frame", "fisheye lens", "from above", "from behind", "from below", "from side", "front view", "full body", "intense angle", "on screen of smartphone camera app frame"]
        lighting_option = ["volumetric lighting", "cinematic lighting", "rim lighting", "dynamic lighting","dimly light", "crepuscular rays", "bokeh"]

        #nothing = ["mating press", "full nelson", "piledriver", "reverse suspended congress", ]
        handjob = ["spitroast", "doggystyle", "amazon position", "legs over head", "reverse cowgirl position", "bent over", "cowgirl position", "girl on top", "missionary"]
        blowjob = ["full nelson", "knees to chest", "doggystyle", "amazon position", "legs over head", "reverse cowgirl position", "bent over", "cowgirl position", "girl on top", "missionary"]
        #bandjob_blowjob = ["amazon position", "legs over head", "spooning"]
        kissing = ["upright straddle", "suspended congress", "reverse upright straddle", "knees to chest", "legs over head", "spooning", "legs over head", "spooning", "boy on top", "bent over", "girl on top", "missionary"]
        grabbing_breast = ["spitroast", "full nelson", "piledriver", "reverse suspended congress", "amazon position", "legs over head", "spooning", "doggystyle", "boy on top", "bent over", "cowgirl position", "missionary"]
        grabbing_hair = ["spitroast", "full nelson", "piledriver", "mating press", "reverse suspended congress", "knees to chest", "legs over head", "spooning", "prone bone", "top-down bottom-up", "doggystyle", "reverse cowgirl position", "boy on top", "bent over", "cowgirl position", "girl on top", "missionary"]
        grabbing_waist = ["spitroast", "mating press", "reverse suspended congress", "amazon position", "doggystyle", "reverse cowgirl position", "boy on top", "bent over", "cowgirl position", "missionary"]
        grabbing_hands = ["prone bone", "top-down bottom-up", "doggystyle", "reverse cowgirl position", "boy on top", "bent over", "cowgirl position", "missionary"]

        if face == "face_string":
            positive_prompt.append(face_string)
        else:
            positive_prompt.append(random.choice(face_option))


        num_of_penis = random.randint(min_number_of_penis, max_number_of_penis)
        if num_of_penis == 1:
            positive_prompt.append("1boy")
            positive_prompt.append("penis")
        elif num_of_penis > 1:
            boys = f"{num_of_penis}boys"
            positive_prompt.append(boys)
            positive_prompt.append("multiple penises")


        if(sex_mode == "sex"):

            if insertion == "deep":
                positive_prompt.append("having a sex, deep insertion")
            elif insertion == "normal":
                positive_prompt.append("having a sex, insertion")
            elif insertion == "imminent":
                positive_prompt.append("imminent sex, imminent insertion")

            sex_option = ["spitroast ", "reverse upright straddle", "mating press", "full nelson", "amazon position", "legs over head", "spooning", "suspended congress", "piledriver", "knees to chest", "reverse suspended congress", "upright straddle", "prone bone", "top-down bottom-up", "doggystyle", "reverse cowgirl position", "boy on top", "bent over", "cowgirl position", "girl on top", "missionary"]
            sex = random.choice(sex_option)
            positive_prompt.append(sex)

            if num_of_penis > 1:
                options = [
                    ["handjob", ""],
                    ["blowjob", ""],
                    ["kissing, forced kiss", "", ""],
                    ["grabbing_breast", "grabbing_breast", ""],
                    ["grabbing_hair", "grabbing_hair", ""],
                    ["grabbing_waist", ""],
                    ["grabbing_hands", ""]
                ]
            else:
                options = [
                    [""],
                    [""],
                    ["kissing, forced kiss", "kissing, forced kiss", ""],
                    ["grabbing_breast", ""],
                    ["grabbing_hair", ""],
                    ["grabbing_waist", ""],
                    ["grabbing_hands", ""]
                ]
            for item in [handjob, blowjob, kissing, grabbing_breast, grabbing_hair, grabbing_waist, grabbing_hands]:
                index = 0
                if(sex in item):
                    item.append("")
                    rst = random.choice(options[index])
                    if rst != "":
                        positive_prompt.append(rst)
                index += 1



        else:
            fellatio = ["fellatio", "irrumatio", "handjob", "penis on face"]
            positive_prompt.append(random.choice(fellatio))

            if insertion == "deep":
                positive_prompt.append("deep throat")
            elif insertion == "normal":
                positive_prompt.append("")
            elif insertion == "imminent":
                positive_prompt.append("imminent fellatio")
            gh = ["grabbing hair", ""]
            if random.choice(gh) == "grabbing hair":
                positive_prompt.append("grabbing hair")






        if focus == "face":
            positive_prompt.append("face forcus, headshot")
        elif focus == "pussy":
            positive_prompt.append("pussy forcus")
        elif focus == "random":
            positive_prompt.append(random.choice(["face forcus, headshot", "pussy forcus"]))

        if topless:
            positive_prompt.append("(topless:1.2)")
        if bottomless:
            positive_prompt.append("(bottomless:1.2)")

        if panty == "random":
            panty = random.choice(["on", "aside", "off"])
        if panty == "on":
            positive_prompt.append("panty")
            negative_prompt.append("pussy")
        elif panty == "aside":
            positive_prompt.append("panty, panty aside")
        elif panty == "off":
            negative_prompt.append("panty")

        if bra == "random":
            bra = random.choice("on", "aside", "off")
        if bra == "on":
            positive_prompt.append("bra")
            negative_prompt.append("nipple")
        elif bra == "aside":
            positive_prompt.append("bra, nipple")
        elif panty == "off":
            negative_prompt.append("bra")

        cum_option = ["cum on face", "cum in mouth", "cum on body", "cum on hair", "cum in pussy"]
        cum_count = random.randint(1, 3)
        cums = []
        if cum == "none":
            pass
        elif cum == "normal":
            cums = random.sample(cum_option, cum_count)
        elif cum == "excessive":
            cums = random.sample(cum_option, cum_count)
            cums.append("excessive cum")
        positive_prompt += cums



        if angle == "angle_string":
            positive_prompt.append(angle_string)
        else:
            positive_prompt.append(random.choice(angle_option))

        if lighting == "lighting_string":
            positive_prompt.append(lighting_string)
        else:
            positive_prompt.append(random.choice(lighting_option))

        print(positive_prompt)
        print(negative_prompt)
        positive_prompt = ",".join(positive_prompt)
        negative_prompt = ",".join(negative_prompt)

        return (positive_prompt, negative_prompt)



class AsunaroImageLoader:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_filename": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "positive_prompt", "negative_prompt")
    FUNCTION = "load_metadata_from_image"

    def load_metadata_from_image(self, image_filename):
        input_dir = folder_paths.get_input_directory()
        full_path = os.path.join(input_dir, image_filename)

        if not os.path.isfile(full_path):
            print(f"File not found: {full_path}")
            return (None, "", "")

        # 画像を開く
        img = Image.open(full_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0  # 0-1に正規化
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # ComfyUIのIMAGE形式に変換

        metadata = img.info

        positive_prompt = metadata.get("asunaro_positive_prompt", "")
        negative_prompt = metadata.get("asunaro_negative_prompt", "")

        print(f"Positive Prompt: {positive_prompt}")
        print(f"Negative Prompt: {negative_prompt}")

        return (img_tensor, positive_prompt, negative_prompt)


class AsunaroBatchImageLoader:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = False

    def __init__(self):
        self.image_files = []  # 画像リスト
        self.current_index = 0  # 現在の画像インデックス
        self.last_mode = None  # 直前のモード（変更時にリセット）

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_name": ("STRING", {"default": "low_res", "multiline": False}),
                "reset_index": ("BOOLEAN", {"default": False}),  # リセット用のフラグ
                "mode": (["sequential", "random"],),  # 画像取得のモード
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  # ランダム時のシード
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "positive_prompt", "negative_prompt", "current_index")
    FUNCTION = "load_next_image"

    def create_empty_image(self, width=512, height=512, message="Reset Completed.\nTurn off the reset switch\n to load images."):
        """
        指定されたサイズの空の画像を生成し、ComfyUIのIMAGE形式で返す。
        - `message` を指定すると、その文字を画像に描画する
            """
        # 空の画像（黒）
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # フォントの設定（Pillowに標準で入っているもの）
        font = ImageFont.load_default(size=24)

        # テキストのサイズを取得（textbboxを使う）
        bbox = draw.textbbox((0, 0), message, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 画像の中央にテキストを配置
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        draw.text((text_x, text_y), message, fill=(255, 255, 255), font=font)

        # 画像を ComfyUI の IMAGE 形式に変換
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # ComfyUIのIMAGE形式
        return img_tensor


    def load_next_image(self, folder_name, reset_index, mode, seed):
        # フォルダパスを取得
        input_dir = folder_paths.get_input_directory()
        folder_path = os.path.join(input_dir, folder_name)

        if not os.path.isdir(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return (self.create_empty_image(), "", "", self.current_index)

        # インデックスのリセット処理
        if reset_index:
            print("🔄 Resetting image index!")
            self.image_files = []
            self.current_index = 0
            return (self.create_empty_image(), "", "", self.current_index)

        # モードが変更された場合もリセット
        if self.last_mode != mode:
            print(f"🔄 Mode changed to {mode}, resetting index!")
            self.image_files = []
            self.current_index = 0
            self.last_mode = mode

        # 画像リストが空なら取得（初回 or リセット後）
        if not self.image_files:
            self.image_files = sorted(
                [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            )
            if not self.image_files:
                print("🚨 No images found in the folder.")
                return (self.create_empty_image(), "", "", self.current_index)

        print(f"📌 Current index is {self.current_index} / {len(self.image_files)}")

        # 画像取得処理
        if mode == "random":
            random.seed(seed)  # シード設定
            self.current_index = random.randint(0, len(self.image_files) - 1)
        else:  # sequential（順番）
            if self.current_index >= len(self.image_files):
                print("✅ All images processed. Resetting index to 0.")
                self.current_index = 0  # 循環

        image_file = self.image_files[self.current_index]
        full_path = os.path.join(folder_path, image_file)

        try:
            img = Image.open(full_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # 画像をテンソル化

            # メタデータ取得
            metadata = img.info
            positive_prompt = metadata.get("asunaro_positive_prompt", "")
            negative_prompt = metadata.get("asunaro_negative_prompt", "")

            print(f"✅ Loaded {image_file}: {positive_prompt}, {negative_prompt}")

            # インデックスを次に進める（sequential の場合のみ）
            if mode == "sequential":
                self.current_index += 1

            return (img_tensor, positive_prompt, negative_prompt, self.current_index)

        except Exception as e:
            print(f"🚨 Failed to load {image_file}: {e}")
            return (self.create_empty_image(), "", "", self.current_index)



class AsunaroSave:
    CATEGORY = "AsuraroTools"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {
                    "multiline": False,
                    "default": "positive_prompt",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": False,
                    "default": "negative_prompt",
                }),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image_with_meta"

    def save_image_with_meta(self, positive_prompt, negative_prompt, images):
        output_dir = folder_paths.get_output_directory()
        # 画像が複数の場合もあるためenumerateで処理する
        for idx, img_tensor in enumerate(images):
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='RGB')

            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("asunaro_positive_prompt", positive_prompt)
            metadata.add_text("asunaro_negative_prompt", negative_prompt)


            dt_now = datetime.datetime.now()
            output_dir = os.path.join(output_dir, "asunaro")
            save_dir = os.path.join(output_dir, dt_now.strftime('%Y%m%d'))
            os.makedirs(save_dir, exist_ok=True)  # フォルダが存在しない場合は作成

            timestamp = dt_now.strftime("%Y%m%d%H%M%S")

            filename = f"{timestamp}_{idx}.png"
            file_path = os.path.join(save_dir, filename)

            pil_img.save(file_path, pnginfo=metadata)
            print(f"Saved image: {file_path}", flush=True)

        return ()


NODE_CLASS_MAPPINGS = {
    "AsunaroWildCard": AsunaroWildCard,
    "AsunaroIfSame": AsunaroIfSame,
    "AsunaroIfContain": AsunaroIfContain,
    "AsunaroIfBiggerThanZero": AsunaroIfBiggerThanZero,
    "AsunaroAnd": AsunaroAnd,
    "AsunaroOr": AsunaroOr,
    "AsunaroIntToStr": AsunaroIntToStr,
    "AsunaroPromptStripper": AsunaroPromptStripper,
    "AsunaroSave": AsunaroSave,
    "AsunaroImageLoader": AsunaroImageLoader,
    "AsunaroBatchImageLoader": AsunaroBatchImageLoader,
    "AsunaroRandomDice": AsunaroRandomDice,
    "AsunaroAutomaticSexPrompter": AsunaroAutomaticSexPrompter,
    "AsunaroTextConcatenator": AsunaroTextConcatenator,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AsunaroWildCard": "AsunaroWildCard",
    "AsunaroIfSame": "AsunaroIfSame",
    "AsunaroIfContain": "AsunaroIfContain",
    "AsunaroIfBiggerThanZero": "AsunaroIfBiggerThanZero",
    "AsunaroAnd": "AsunaroAnd",
    "AsunaroOr": "AsunaroOr",
    "AsunaroIntToStr": "AsunaroIntToStr",
    "AsunaroPromptStripper": "AsunaroPromptStripper",
    "AsunaroSave": "AsunaroSave",
    "AsunaroImageLoader": "AsunaroImageLoader",
    "AsunaroBatchImageLoader": "AsunaroBatchImageLoader",
    "AsunaroRandomDice": "AsunaroRandomDice",
    "AsunaroAutomaticSexPrompter": "AsunaroAutomaticSexPrompter",
    "AsunaroTextConcatenator": "AsunaroTextConcatenator",
}

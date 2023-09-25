import os

from clip_interrogator import Config, Interrogator, LabelTable, load_list

DATA_PATH = './assets/clip2/data'


class InterrogateModels:
    model = None
    config = None

    def __init__(self):
        self.config = Config(
            clip_model_name="ViT-L-14/openai",
            clip_model_path="./assets/clip2/model",
            data_path="./assets/clip2/data",
            cache_path="./assets/clip2/cache",
        )
        self.config.apply_low_vram_defaults()

    def load(self):
        self.model = Interrogator(self.config)

    def unload(self):
        pass

    def generate_caption(self, pil_image):
        return self.interrogate(pil_image, stringify=True)

    def interrogate(self, pil_image, stringify=False):
        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble',
                 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount',
                 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on " + site for site in sites])
        trending_list.extend(["featured on " + site for site in sites])
        trending_list.extend([site + " contest winner" for site in sites])
        artists = LabelTable(load_list(DATA_PATH, 'artists.txt'), "artists", self.model)
        flavors = LabelTable(load_list(DATA_PATH, 'flavors.txt'), "flavors", self.model)
        mediums = LabelTable(load_list(DATA_PATH, 'mediums.txt'), "mediums", self.model)
        movements = LabelTable(load_list(DATA_PATH, 'movements.txt'), "movements", self.model)
        trendings = LabelTable(trending_list, "trendings", self.model)
        tables = [artists, flavors, mediums, movements, trendings]
        results = []
        for table in tables:
            feat = self.model.image_to_features(pil_image)
            match_results = table.rank(feat, top_count=5)
            ranks = self.model.similarities(feat, match_results)
            for i in range(len(match_results)):
                results.append({
                    "tag": match_results[i],
                    "rank": ranks[i],
                })
        if stringify:
            return ",".join([result["tag"] for result in results])
        else:
            return results


model = InterrogateModels()

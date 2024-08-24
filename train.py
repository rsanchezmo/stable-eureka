from stable_eureka.flexibility_wrapper import FlexibleStableEureka


if __name__ == '__main__':
    trainer = FlexibleStableEureka(config_path='./configs/bipedal_walker.yml')
    trainer.run(verbose=True)
